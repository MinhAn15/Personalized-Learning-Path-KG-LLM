from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional

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
