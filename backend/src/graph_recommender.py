from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from .config import Config
from .data_loader import execute_cypher_query

logger = logging.getLogger(__name__)

try:  # Optional deep learning stack
    import torch  # type: ignore
    from torch import nn  # type: ignore
    import torch.nn.functional as F  # type: ignore
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore

try:  # PyG components
    from torch_geometric.data import Data  # type: ignore
    from torch_geometric.nn import GATConv, GCNConv  # type: ignore
except ImportError:  # pragma: no cover
    Data = None  # type: ignore
    GATConv = None  # type: ignore
    GCNConv = None  # type: ignore


def _normalize(value: float, max_value: float) -> float:
    if max_value <= 0.0:
        return 0.0
    return float(min(max(value / max_value, 0.0), 1.0))


def _difficulty_to_score(label: str) -> float:
    if not label:
        return 0.5
    label = label.upper()
    mapping = {
        "BEGINNER": 0.2,
        "INTRO": 0.3,
        "STANDARD": 0.5,
        "INTERMEDIATE": 0.6,
        "ADVANCED": 0.8,
        "EXPERT": 1.0,
    }
    return mapping.get(label, 0.5)


@dataclass
class GraphRecommenderConfig:
    embedding_dim: int = 64
    hidden_dim: int = 64
    epochs: int = 40
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    negative_ratio: int = 2
    min_interactions: int = 8
    max_samples: int = 50_000
    seed: int = 42


@dataclass
class _GraphArtifacts:
    data: Optional[Any]
    student_ids: List[str]
    concept_ids: List[str]
    student_index: Dict[str, int]
    concept_index: Dict[str, int]
    interactions: List[Tuple[int, int, float]]
    student_history: Dict[str, List[str]]
    student_stats: Dict[str, Dict[str, float]]
    concept_metadata: Dict[str, Dict[str, Any]]
    concept_frequency: Dict[str, int]

    @property
    def concept_offset(self) -> int:
        return len(self.student_ids)


class _GraphHybridNet(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int, embedding_dim: int) -> None:
        super().__init__()
        self.gcn = GCNConv(in_features, hidden_dim)  # type: ignore[arg-type]
        self.gat = GATConv(hidden_dim, hidden_dim, heads=2, concat=False)  # type: ignore[arg-type]
        self.projection = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:  # type: ignore[name-defined]
        h = self.gcn(x, edge_index)
        h = torch.relu(h)
        h = self.gat(h, edge_index)
        h = torch.relu(h)
        return self.projection(h)


class GraphRecommender:
    def __init__(self, neo4j_driver: Any | None = None, config: Optional[GraphRecommenderConfig] = None) -> None:
        self.driver = neo4j_driver
        self.config = config or GraphRecommenderConfig()
        self._graph: Optional[_GraphArtifacts] = None
        self._model: Optional[_GraphHybridNet] = None
        self._embeddings: Optional[torch.Tensor] = None  # type: ignore[name-defined]
        self._trained = False
        self._gnn_ready = bool(torch and nn and F and Data and GATConv and GCNConv)
        if torch is not None:
            random.seed(self.config.seed)
            np.random.seed(self.config.seed)
            torch.manual_seed(self.config.seed)
        if torch is not None and torch.cuda.is_available():  # type: ignore[union-attr]
            torch.cuda.manual_seed_all(self.config.seed)  # type: ignore[attr-defined]
        self.device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def train(self, force: bool = False) -> None:
        if not self._gnn_ready:
            logger.info("Graph recommender running in fallback mode (PyTorch Geometric missing).")
            return
        graph = self._ensure_graph()
        if not graph or not graph.data:
            logger.info("Graph recommender has insufficient data to train.")
            return
        if self._trained and not force:
            return
        self._train_model(graph)
        self._trained = True

    def recommend_next_concept(self, learner_id: str, top_k: int = 5) -> Dict[str, Any]:
        graph = self._ensure_graph()
        if not graph:
            return {"mode": "fallback", "recommendations": []}
        top_k = max(1, min(int(top_k or 5), 50))
        if not self._gnn_ready or not graph.data:
            recs = self._fallback_recommendations(graph, learner_id, top_k)
            return {"mode": "fallback", "recommendations": recs}
        if not self._trained or self._embeddings is None:
            self.train()
        if self._embeddings is None:
            recs = self._fallback_recommendations(graph, learner_id, top_k)
            return {"mode": "fallback", "recommendations": recs}
        if learner_id not in graph.student_index:
            recs = self._fallback_recommendations(graph, learner_id, top_k)
            return {"mode": "fallback", "recommendations": recs}

        student_idx = graph.student_index[learner_id]
        student_vec = self._embeddings[student_idx]
        concept_offset = graph.concept_offset
        concept_vecs = self._embeddings[concept_offset: concept_offset + len(graph.concept_ids)]
        with torch.no_grad():  # type: ignore[attr-defined]
            scores = torch.sigmoid(torch.einsum("d,nd->n", student_vec, concept_vecs))  # type: ignore[attr-defined]
        learned = set(graph.student_history.get(learner_id, []))
        ranked: List[Tuple[str, float]] = []
        for idx, concept_id in enumerate(graph.concept_ids):
            if concept_id in learned:
                continue
            ranked.append((concept_id, float(scores[idx].item())))
        ranked.sort(key=lambda item: item[1], reverse=True)
        result = []
        for concept_id, score in ranked[:top_k]:
            meta = graph.concept_metadata.get(concept_id, {})
            result.append(
                {
                    "concept_id": concept_id,
                    "score": score,
                    "priority": meta.get("priority"),
                    "difficulty": meta.get("difficulty"),
                    "title": meta.get("concept"),
                }
            )
        if not result:
            return {"mode": "gnn", "recommendations": self._fallback_recommendations(graph, learner_id, top_k)}
        return {"mode": "gnn", "recommendations": result}

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------
    def _ensure_graph(self) -> Optional[_GraphArtifacts]:
        if self._graph is not None:
            return self._graph
        if not self.driver:
            logger.warning("Graph recommender requires Neo4j driver for initialization.")
            return None
        interactions = self._load_interactions()
        if len(interactions) < self.config.min_interactions:
            logger.info("Not enough interactions to build graph (count=%d).", len(interactions))
            self._graph = _GraphArtifacts(
                data=None,
                student_ids=[],
                concept_ids=[],
                student_index={},
                concept_index={},
                interactions=[],
                student_history={},
                student_stats={},
                concept_metadata={},
                concept_frequency={},
            )
            return self._graph
        graph = self._build_graph(interactions)
        self._graph = graph
        return graph

    def _load_interactions(self) -> List[Dict[str, Any]]:
        query = f"""
        MATCH (s:Student)-[:HAS_LEARNING_DATA]->(ld:LearningData)
        WITH s, ld
        MATCH (n:KnowledgeNode {{{Config.PROPERTY_ID}: ld.node_id}})
        RETURN s.StudentID AS student_id,
               ld.node_id AS concept_id,
               coalesce(toFloat(ld.score), 0.0) AS score,
               coalesce(ld.time_spent, 0) AS time_spent,
               coalesce(n.{Config.PROPERTY_PRIORITY}, 0.0) AS priority,
               coalesce(n.{Config.PROPERTY_DIFFICULTY}, 'STANDARD') AS difficulty,
               coalesce(n.{Config.PROPERTY_SANITIZED_CONCEPT}, n.{Config.PROPERTY_LEARNING_OBJECTIVE}) AS concept
        ORDER BY ld.timestamp ASC
        LIMIT $limit
        """
        try:
            rows = execute_cypher_query(self.driver, query, {"limit": self.config.max_samples})
        except Exception as exc:  # pragma: no cover
            logger.warning("Graph recommender failed to fetch interactions: %s", exc)
            return []
        return rows or []

    def _build_graph(self, interactions: List[Dict[str, Any]]) -> _GraphArtifacts:
        student_history: Dict[str, List[str]] = {}
        student_scores: Dict[str, List[float]] = {}
        student_time: Dict[str, List[float]] = {}
        concept_frequency: Dict[str, int] = {}
        concept_metadata: Dict[str, Dict[str, Any]] = {}

        for row in interactions:
            student_id = str(row.get("student_id") or "").strip()
            concept_id = str(row.get("concept_id") or "").strip()
            if not student_id or not concept_id:
                continue
            student_history.setdefault(student_id, []).append(concept_id)
            score = float(row.get("score") or 0.0)
            time_spent = float(row.get("time_spent") or 0.0)
            student_scores.setdefault(student_id, []).append(score)
            student_time.setdefault(student_id, []).append(time_spent)
            concept_frequency[concept_id] = concept_frequency.get(concept_id, 0) + 1
            if concept_id not in concept_metadata:
                concept_metadata[concept_id] = {
                    "priority": float(row.get("priority") or 0.0),
                    "difficulty": row.get("difficulty") or "STANDARD",
                    "concept": row.get("concept"),
                }

        student_ids = sorted(student_history.keys())
        concept_ids = sorted(concept_frequency.keys())
        student_index = {sid: idx for idx, sid in enumerate(student_ids)}
        concept_index = {cid: idx + len(student_ids) for idx, cid in enumerate(concept_ids)}

        features: List[List[float]] = []
        max_sessions = max((len(v) for v in student_history.values()), default=1)
        for sid in student_ids:
            scores = student_scores.get(sid, [0.0])
            times = student_time.get(sid, [0.0])
            avg_score = np.clip(np.mean(scores) / 100.0, 0.0, 1.0)
            avg_time = np.clip(np.mean(times) / 60.0, 0.0, 1.0)
            session_norm = _normalize(len(student_history[sid]), float(max_sessions))
            features.append([1.0, 0.0, avg_score, avg_time, session_norm])

        all_priorities = [meta.get("priority", 0.0) for meta in concept_metadata.values()]
        max_priority = max(all_priorities) if all_priorities else 1.0
        for cid in concept_ids:
            meta = concept_metadata.get(cid, {})
            priority_norm = _normalize(float(meta.get("priority") or 0.0), float(max_priority))
            difficulty_score = _difficulty_to_score(str(meta.get("difficulty") or "STANDARD"))
            popularity = _normalize(float(concept_frequency.get(cid, 1)), float(max(concept_frequency.values()) or 1))
            features.append([0.0, 1.0, priority_norm, difficulty_score, popularity])

        edge_pairs: List[Tuple[int, int]] = []
        weighted_interactions: List[Tuple[int, int, float]] = []
        for row in interactions:
            sid = str(row.get("student_id") or "").strip()
            cid = str(row.get("concept_id") or "").strip()
            if sid not in student_index or cid not in concept_index:
                continue
            s_idx = student_index[sid]
            c_idx = concept_index[cid]
            edge_pairs.append((s_idx, c_idx))
            edge_pairs.append((c_idx, s_idx))
            weight = float(row.get("score") or 0.0) / 100.0
            weighted_interactions.append((s_idx, c_idx, weight))

        if not edge_pairs:
            data = None
        elif not self._gnn_ready:
            data = None
        else:
            edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()  # type: ignore[attr-defined]
            x = torch.tensor(np.asarray(features, dtype=np.float32), dtype=torch.float32)  # type: ignore[attr-defined]
            data = Data(x=x, edge_index=edge_index)

        student_stats = {}
        for sid in student_ids:
            stats = {
                "avg_score": float(np.mean(student_scores.get(sid, [0.0]))),
                "session_count": len(student_history.get(sid, [])),
                "avg_minutes": float(np.mean(student_time.get(sid, [0.0])) / 60.0),
            }
            student_stats[sid] = stats

        return _GraphArtifacts(
            data=data,
            student_ids=student_ids,
            concept_ids=concept_ids,
            student_index=student_index,
            concept_index=concept_index,
            interactions=weighted_interactions,
            student_history={sid: list(history) for sid, history in student_history.items()},
            student_stats=student_stats,
            concept_metadata=concept_metadata,
            concept_frequency=concept_frequency,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def _train_model(self, graph: _GraphArtifacts) -> None:
        assert graph.data is not None
        assert torch is not None and nn is not None and F is not None
        model = _GraphHybridNet(graph.data.num_node_features, self.config.hidden_dim, self.config.embedding_dim)
        model.to(self.device)
        data = graph.data.to(self.device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        positives_students = torch.tensor([pair[0] for pair in graph.interactions], dtype=torch.long, device=self.device)
        positives_concepts = torch.tensor([pair[1] for pair in graph.interactions], dtype=torch.long, device=self.device)
        if positives_students.numel() == 0:
            logger.info("Graph recommender cannot train without positive interactions.")
            return

        concept_offset = graph.concept_offset
        concept_count = len(graph.concept_ids)
        if concept_count == 0:
            logger.info("Graph recommender has no concept nodes to optimize over.")
            return

        for epoch in range(self.config.epochs):
            model.train()
            optimizer.zero_grad()
            embeddings = model(data.x, data.edge_index)
            student_vecs = embeddings[positives_students]
            concept_vecs = embeddings[positives_concepts]
            pos_logits = (student_vecs * concept_vecs).sum(dim=1)
            neg_samples = positives_students.repeat_interleave(self.config.negative_ratio)
            neg_concepts = torch.randint(0, concept_count, (neg_samples.size(0),), device=self.device)
            neg_indices = neg_concepts + concept_offset
            neg_student_vecs = embeddings[neg_samples]
            neg_concept_vecs = embeddings[neg_indices]
            neg_logits = (neg_student_vecs * neg_concept_vecs).sum(dim=1)
            pos_loss = F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits))
            neg_loss = F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits))
            loss = pos_loss + neg_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if (epoch + 1) % 10 == 0:
                logger.debug("Graph recommender epoch %d loss %.4f", epoch + 1, float(loss.item()))

        model.eval()
        with torch.no_grad():
            embeddings = model(data.x, data.edge_index).detach().cpu()
        self._model = model
        self._embeddings = embeddings

    # ------------------------------------------------------------------
    # Fallback logic
    # ------------------------------------------------------------------
    def _fallback_recommendations(
        self,
        graph: _GraphArtifacts,
        learner_id: str,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        learned = set(graph.student_history.get(learner_id, []))
        recommendations: List[Tuple[str, float]] = []
        for concept_id, meta in graph.concept_metadata.items():
            if concept_id in learned:
                continue
            freq = float(graph.concept_frequency.get(concept_id, 0))
            priority = float(meta.get("priority") or 0.0)
            score = priority + freq
            recommendations.append((concept_id, score))
        recommendations.sort(key=lambda item: item[1], reverse=True)
        result = []
        for concept_id, score in recommendations[:top_k]:
            meta = graph.concept_metadata.get(concept_id, {})
            result.append(
                {
                    "concept_id": concept_id,
                    "score": float(score),
                    "priority": meta.get("priority"),
                    "difficulty": meta.get("difficulty"),
                    "title": meta.get("concept"),
                }
            )
        return result

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def as_dict(self) -> Dict[str, Any]:
        graph = self._graph
        return {
            "trained": self._trained,
            "gnn_ready": self._gnn_ready,
            "students": len(graph.student_ids) if graph else 0,
            "concepts": len(graph.concept_ids) if graph else 0,
        }
