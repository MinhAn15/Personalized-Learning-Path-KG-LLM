from __future__ import annotations

import logging
import math
import random
from datetime import datetime, timedelta
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

try:
    import torch  # type: ignore
    from torch import nn  # type: ignore
    import torch.nn.functional as F  # type: ignore
except ImportError:  # pragma: no cover - handled gracefully at runtime
    torch = None
    nn = None
    F = None

from .neo4j_manager import Neo4jManager

logger = logging.getLogger(__name__)


class AdvancedKnowledgeTracer:
    """
    Knowledge tracing adapted to current schema:
    - Student (StudentID)
    - KnowledgeNode (Node_ID)
    - LearningData events linked via (Student)-[:HAS_LEARNING_DATA]->(ld)-[:RELATED_TO_NODE]->(KnowledgeNode)

    Provides temporal decay and simple Bayesian-like update.
    """

    FORGETTING_INDEX = 0.5  # Ebbinghaus-like intensity

    def __init__(self, neo4j: Neo4jManager):
        self.neo4j = neo4j

    def _fetch_ld(self, student_id: str, node_id: str) -> List[Dict]:
        q = """
        MATCH (s:Student {StudentID: $sid})-[:HAS_LEARNING_DATA]->(ld:LearningData)-[:RELATED_TO_NODE]->(n:KnowledgeNode {Node_ID: $nid})
        RETURN ld.timestamp AS ts, toFloat(ld.score) AS score, toInteger(ld.time_spent) AS time_spent
        ORDER BY ts ASC
        """
        return self.neo4j.execute_read(q, {"sid": student_id, "nid": node_id})

    def compute_mastery_with_decay(self, student_id: str, node_id: str) -> Dict[str, Any]:
        """Compute mastery score with temporal decay based on LearningData."""
        rows = self._fetch_ld(student_id, node_id)
        if not rows:
            return {
                "score": 0.0,
                "confidence": 0.0,
                "status": "not_started",
                "next_review_date": datetime.now(),
            }

        # Last assessment approximated as the last ld timestamp
        last_ts = rows[-1]["ts"]
        try:
            last_assessed = (
                datetime.fromisoformat(last_ts) if isinstance(last_ts, str) else last_ts
            )
        except Exception:
            last_assessed = datetime.now()

        # Weighted by recency (log-like decay)
        scores = [float(r.get("score") or 0.0) for r in rows]
        times = [max(0, int(r.get("time_spent") or 0)) for r in rows]
        # normalize scores to [0,1] if they look like percentages
        if max(scores) > 1.0:
            scores = [min(1.0, s / 100.0) for s in scores]

        decayed = 0.0
        denom = 0.0
        n = len(scores)
        for idx, s in enumerate(scores):
            # more recent -> larger weight
            recency = 1 - self.FORGETTING_INDEX * math.log(1 + (n - 1 - idx))
            recency = max(0.1, recency)
            decayed += s * recency
            denom += recency
        score = decayed / denom if denom else 0.0
        score = max(0.0, min(1.0, score))

        status = self._classify_mastery_status(score)
        next_review = self._calculate_optimal_review_date(score, last_assessed, status)

        # Confidence: heuristic using variability and count
        try:
            variability = pstdev(scores) if len(scores) > 1 else 0.0
        except Exception:
            variability = 0.0
        confidence = max(0.0, min(1.0, 0.6 + 0.4 * (1 - variability)))

        return {
            "score": score,
            "confidence": confidence,
            "status": status,
            "days_since_assessment": (datetime.now() - last_assessed).days,
            "next_review_date": next_review,
            "recent_scores": scores[-5:],
            "avg_score": mean(scores) if scores else 0.0,
            "interactions": len(scores),
        }

    def update_mastery_after_assessment(
        self,
        student_id: str,
        node_id: str,
        performance_score: float,
        assessment_method: str = "quiz",
    ) -> Dict:
        """Merge/Update a MASTERY relationship with posterior score and metadata."""
        # prior from compute
        prior = self.compute_mastery_with_decay(student_id, node_id)
        prior_score = prior.get("score", 0.0)
        perf = performance_score
        perf01 = perf if perf <= 1.0 else perf / 100.0

        likelihood = 0.9 if perf01 > 0.8 else 0.5 if perf01 > 0.6 else 0.2
        posterior = (likelihood * prior_score) / (
            likelihood * prior_score + (1 - likelihood) * (1 - prior_score)
        ) if (likelihood * prior_score + (1 - likelihood) * (1 - prior_score)) > 0 else prior_score
        posterior = max(0.0, min(1.0, posterior))

        status = self._classify_mastery_status(posterior)
        next_review = self._calculate_optimal_review_date(posterior, datetime.now(), status)

        q = """
        MATCH (s:Student {StudentID: $sid}), (n:KnowledgeNode {Node_ID: $nid})
        MERGE (s)-[m:MASTERY]->(n)
        SET m.previousScore = coalesce(m.score, 0.0),
            m.score = $score,
            m.confidence = $confidence,
            m.lastAssessmentDate = datetime(),
            m.assessmentMethod = $method,
            m.nextOptimalReviewDate = $next_review
        RETURN m
        """
        params = {
            "sid": student_id,
            "nid": node_id,
            "score": posterior,
            "confidence": min(1.0, (prior.get("confidence", 0.5) or 0.5) + 0.1),
            "method": assessment_method,
            "next_review": next_review.isoformat(),
        }
        return self.neo4j.execute_write(q, params)

    # --- Helpers --------------------------------------------------------------
    @staticmethod
    def _classify_mastery_status(score: float) -> str:
        if score < 0.3:
            return "not_mastered"
        if score < 0.7:
            return "partial_mastery"
        if score < 0.95:
            return "mastered"
        return "expert"

    @staticmethod
    def _calculate_optimal_review_date(score: float, last_assessed: datetime, status: str) -> datetime:
        if status == "not_mastered":
            days = 1
        elif status == "partial_mastery":
            days = 3 if score < 0.5 else 7
        elif status == "mastered":
            days = 14
        else:
            days = 30
        return (last_assessed or datetime.now()) + timedelta(days=days)


if nn is not None:

    class _DKTModel(nn.Module):
        """Minimal LSTM-based DKT network."""

        def __init__(self, input_dim: int, hidden_size: int, num_skills: int):
            super().__init__()
            self.hidden_size = hidden_size
            self.input_layer = nn.Linear(input_dim, hidden_size)
            self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
            self.output_layer = nn.Linear(hidden_size, num_skills)

    def forward(self, x: Any) -> Any:
            h = torch.relu(self.input_layer(x))
            out, _ = self.lstm(h)
            logits = self.output_layer(out)
            return torch.sigmoid(logits)

else:  # pragma: no cover - fallback for typing when torch absent

    class _DKTModel:  # type: ignore[empty-body]
        pass


class DeepKnowledgeTracer:
    """Deep Knowledge Tracing using a lightweight PyTorch LSTM implementation."""

    def __init__(
        self,
        neo4j: Neo4jManager,
        hidden_size: int = 64,
        learning_rate: float = 1e-3,
        score_threshold: float = 0.7,
        max_epochs: int = 8,
        device: Optional[str] = None,
    ) -> None:
        if torch is None:
            raise ImportError(
                "DeepKnowledgeTracer requires PyTorch. Please install torch in the backend environment."
            )

        self.neo4j = neo4j
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.score_threshold = score_threshold
        self.max_epochs = max_epochs
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model: Optional[_DKTModel] = None
        self.num_skills: int = 0
        self.input_dim: int = 0
        self.skill_to_index: Dict[str, int] = {}
        self.index_to_skill: Dict[int, str] = {}
        self.student_sequences: Dict[str, List[Tuple[int, int]]] = {}
        self.dataset_df: pd.DataFrame = pd.DataFrame()
        self.trained: bool = False

    # --- Public API -------------------------------------------------------
    def ensure_trained(self) -> None:
        if not self.trained:
            trained = self.train(self.max_epochs)
            if not trained:
                raise RuntimeError("DeepKnowledgeTracer training aborted: insufficient data.")

    def train(self, epochs: int = 8) -> bool:
        df = self._load_learning_dataframe()
        sequences = self._prepare_sequences(df)
        if not sequences or self.num_skills == 0:
            logger.warning("DKT: No sequences available for training.")
            return False

        self.model = _DKTModel(self.input_dim, self.hidden_size, self.num_skills).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            batches = 0
            for seq in sequences:
                tensors = self._sequence_to_training_tensors(seq)
                if tensors is None:
                    continue
                inputs, target_skills, target_correct = tensors
                optimizer.zero_grad()
                preds = self.model(inputs)
                preds = preds.squeeze(0)  # (seq_len, num_skills)
                selected = preds.gather(1, target_skills.unsqueeze(1)).squeeze(1)
                loss = F.binary_cross_entropy(selected, target_correct)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item())
                batches += 1

            if batches:
                avg_loss = total_loss / batches
                logger.debug("DKT epoch %d | loss=%.4f", epoch + 1, avg_loss)

        self.model.eval()
        self.trained = True
        self.dataset_df = df
        return True

    def predict_mastery(self, student_id: str, node_id: str) -> float:
        self.ensure_trained()
        if node_id not in self.skill_to_index:
            logger.debug("DKT: Node %s unseen during training.", node_id)
            return 0.0
        seq = self.student_sequences.get(student_id)
        if not seq:
            logger.debug("DKT: No history for student %s.", student_id)
            return 0.0

        inputs = self._sequence_to_prediction_tensor(seq)
        if inputs is None or self.model is None:
            return 0.0

        with torch.no_grad():
            preds = self.model(inputs)
        skill_idx = self.skill_to_index[node_id]
        prob = preds[0, -1, skill_idx].item()
        return float(max(0.0, min(1.0, prob)))

    def update_mastery_dkt(self, student_id: str, node_id: str) -> Dict[str, Any]:
        prob = self.predict_mastery(student_id, node_id)
        status = AdvancedKnowledgeTracer._classify_mastery_status(prob)
        query = """
        MATCH (s:Student {StudentID: $sid}), (n:KnowledgeNode {Node_ID: $nid})
        MERGE (s)-[m:MASTERY]->(n)
        SET m.score = $score,
            m.model = 'dkt',
            m.lastDKTUpdate = datetime()
        RETURN m.score AS score, m.model AS model, m.lastDKTUpdate AS updated_at
        """
        params = {"sid": student_id, "nid": node_id, "score": prob}
        result = self.neo4j.execute_write(query, params)
        return {
            "student_id": student_id,
            "node_id": node_id,
            "predicted_mastery": prob,
            "status": status,
            "neo4j_response": result,
        }

    # --- Data preparation --------------------------------------------------
    def _load_learning_dataframe(self) -> pd.DataFrame:
        query = """
        MATCH (s:Student)-[:HAS_LEARNING_DATA]->(ld:LearningData)-[:RELATED_TO_NODE]->(n:KnowledgeNode)
        RETURN s.StudentID AS student_id,
               n.Node_ID AS node_id,
               toFloat(ld.score) AS raw_score,
               ld.timestamp AS timestamp
        ORDER BY student_id, timestamp
        """
        try:
            rows = self.neo4j.execute_read(query)
            df = pd.DataFrame(rows)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("DKT: failed to load learning data from Neo4j: %s", exc)
            df = pd.DataFrame()

        if df.empty:
            logger.info("DKT: using mock LearningData dataframe for training.")
            df = self._mock_learning_dataframe()

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["student_id", "node_id", "raw_score"])
        df = df.sort_values(["student_id", "timestamp"]).reset_index(drop=True)
        df["score"] = df["raw_score"].apply(self._normalize_score)
        df["correct"] = df["score"].apply(lambda s: 1 if s >= self.score_threshold else 0)
        return df[["student_id", "node_id", "score", "correct", "timestamp"]]

    def _prepare_sequences(self, df: pd.DataFrame) -> List[List[Tuple[int, int]]]:
        nodes = sorted(df["node_id"].unique())
        self.skill_to_index = {node: idx for idx, node in enumerate(nodes)}
        self.index_to_skill = {idx: node for node, idx in self.skill_to_index.items()}
        self.num_skills = len(self.skill_to_index)
        self.input_dim = self.num_skills * 2

        sequences: Dict[str, List[Tuple[int, int]]] = {}
        for student_id, group in df.groupby("student_id"):
            history: List[Tuple[int, int]] = []
            for _, row in group.iterrows():
                skill_idx = self.skill_to_index.get(row["node_id"])
                if skill_idx is None:
                    continue
                history.append((skill_idx, int(row["correct"])))
            if history:
                sequences[student_id] = history

        self.student_sequences = sequences
        return list(sequences.values())

    # --- Tensor utilities --------------------------------------------------
    def _sequence_to_training_tensors(
        self, seq: List[Tuple[int, int]]
    ) -> Optional[Tuple[Any, Any, Any]]:
        if len(seq) < 2 or self.input_dim == 0:
            return None

        input_vectors: List[List[float]] = []
        target_skills: List[int] = []
        target_correct: List[float] = []

        for idx in range(len(seq) - 1):
            skill, correct = seq[idx]
            next_skill, next_correct = seq[idx + 1]
            input_vectors.append(self._interaction_vector(skill, correct))
            target_skills.append(next_skill)
            target_correct.append(float(next_correct))

        inputs = torch.tensor(input_vectors, dtype=torch.float32, device=self.device).unsqueeze(0)
        target_skill_tensor = torch.tensor(target_skills, dtype=torch.long, device=self.device)
        target_correct_tensor = torch.tensor(target_correct, dtype=torch.float32, device=self.device)
        return inputs, target_skill_tensor, target_correct_tensor

    def _sequence_to_prediction_tensor(self, seq: List[Tuple[int, int]]) -> Optional[Any]:
        if not seq or self.input_dim == 0:
            return None
        vectors = [self._interaction_vector(skill, correct) for skill, correct in seq]
        tensor = torch.tensor(vectors, dtype=torch.float32, device=self.device).unsqueeze(0)
        return tensor

    def _interaction_vector(self, skill_idx: int, correct: int) -> List[float]:
        vec = [0.0] * self.input_dim
        offset = self.num_skills if correct >= 1 else 0
        vec[skill_idx + offset] = 1.0
        return vec

    @staticmethod
    def _normalize_score(score: Any) -> float:
        try:
            value = float(score)
        except (TypeError, ValueError):
            return 0.0
        if value > 1.0:
            value /= 100.0
        return max(0.0, min(1.0, value))

    @staticmethod
    def _mock_learning_dataframe() -> pd.DataFrame:
        base_time = datetime.utcnow() - timedelta(days=14)
        students = ["mock_student_1", "mock_student_2", "mock_student_3"]
        nodes = ["concept_101", "concept_102", "concept_201", "concept_202", "concept_301"]
        entries: List[Dict[str, Any]] = []
        random.seed(42)
        for student in students:
            current = base_time
            for _ in range(12):
                node = random.choice(nodes)
                score = max(0.0, min(1.0, random.gauss(0.7, 0.15)))
                entries.append(
                    {
                        "student_id": student,
                        "node_id": node,
                        "score": score,
                        "correct": 1 if score >= 0.7 else 0,
                        "timestamp": current,
                    }
                )
                current += timedelta(hours=random.randint(6, 18))
        return pd.DataFrame(entries)


def update_mastery_dkt(
    neo4j: Neo4jManager,
    student_id: str,
    node_id: str,
    tracer: Optional[DeepKnowledgeTracer] = None,
) -> Dict[str, Any]:
    """Convenience wrapper to train (if needed) and update mastery with DKT."""

    tracer = tracer or DeepKnowledgeTracer(neo4j)
    return tracer.update_mastery_dkt(student_id, node_id)
