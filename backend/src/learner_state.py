from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Any
from datetime import date, timedelta

from .config import Config


@dataclass
class LearnerState:
    """
    Encapsulates a student's evolving state for adaptive planning.

    Fields
    - student_id: Unique id of the learner
    - mastery: concept_id -> mastery score in [0, 1]
    - cognitive_level_goal: concept_id -> Bloom level (REMEMBER..CREATE)
    - time_budget_minutes: optional planning budget
    - preferences: user preferences or constraints (e.g., modality, pace)

    Methods are intentionally lightweight and side-effect free (except update_mastery).
    They shouldn't depend on external services; callers provide inputs.
    """

    student_id: str
    mastery: Dict[str, float] = field(default_factory=dict)
    cognitive_level_goal: Dict[str, str] = field(default_factory=dict)
    time_budget_minutes: Optional[int] = None
    preferences: Dict[str, Any] = field(default_factory=dict)

    def get_mastery(self, concept_id: str, default: float = 0.0) -> float:
        m = self.mastery.get(concept_id, default)
        return max(0.0, min(1.0, m))

    def in_zpd(
        self,
        concept_id: str,
        lower: Optional[float] = None,
        upper: Optional[float] = None,
    ) -> bool:
        """Return True if concept mastery is within Zone of Proximal Development bounds."""
        lo = Config.ZPD_DEFAULT_LOWER if lower is None else lower
        hi = Config.ZPD_DEFAULT_UPPER if upper is None else upper
        m = self.get_mastery(concept_id)
        return lo <= m <= hi

    def update_mastery(
        self,
        concept_id: str,
        score: float,
        time_spent_minutes: Optional[float] = None,
        attempts: int = 1,
        weight: Optional[float] = None,
    ) -> float:
        """
        Update mastery with a bounded exponential moving average.

        Inputs
        - score: performance in [0,1]
        - time_spent_minutes: optional; long time with low score penalizes update
        - attempts: >=1, dampens update when many attempts required
        - weight: override smoothing factor (alpha). Defaults to 0.35 scaled by attempts/time.

        Returns the new mastery value in [0,1].
        """
        score = max(0.0, min(1.0, score))
        prev = self.get_mastery(concept_id, 0.0)

        # Base alpha and adaptive damping
        alpha = 0.35 if weight is None else max(0.0, min(1.0, weight))
        # Attempts damping: more attempts -> smaller update
        attempt_factor = 1.0 / max(1.0, float(attempts))
        # Time damping: if took much longer than “typical” 15m, shrink update
        if time_spent_minutes is not None:
            time_factor = 15.0 / max(5.0, time_spent_minutes)
            time_factor = max(0.25, min(1.25, time_factor))
        else:
            time_factor = 1.0
        a = max(0.05, min(0.95, alpha * attempt_factor * time_factor))

        new_mastery = (1.0 - a) * prev + a * score
        new_mastery = max(0.0, min(1.0, new_mastery))
        self.mastery[concept_id] = new_mastery
        return new_mastery

    def estimate_next_review(self, concept_id: str, today: Optional[date] = None) -> date:
        """
        Estimate next review date based on a simple forgetting curve approximation.
        Higher mastery lengthens interval; bounded by REVIEW_MIN_DAYS..REVIEW_MAX_DAYS.
        """
        if today is None:
            today = date.today()
        m = self.get_mastery(concept_id, 0.0)
        # Map mastery to [min,max] days with slight convexity
        span = Config.REVIEW_MAX_DAYS - Config.REVIEW_MIN_DAYS
        days = Config.REVIEW_MIN_DAYS + span * (m ** 1.2)
        return today + timedelta(days=int(round(days)))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "student_id": self.student_id,
            "mastery": dict(self.mastery),
            "cognitive_level_goal": dict(self.cognitive_level_goal),
            "time_budget_minutes": self.time_budget_minutes,
            "preferences": dict(self.preferences),
        }

    # --- Factories & computations -------------------------------------------
    @classmethod
    def from_neo4j(cls, driver: Any, student_id: str) -> "LearnerState":
        """
        Build a LearnerState from Neo4j Student and LearningData nodes.
        Mastery is computed per concept using a simple recency-decayed EMA of scores.
        """
        # Load student profile basics
        from .data_loader import execute_cypher_query

        # Preferences and time budget (if stored on Student)
        srow = execute_cypher_query(
            driver,
            """
            MATCH (s:Student {StudentID: $sid})
            RETURN s.learning_style_preference AS style,
                   s.time_availability AS time_availability
            """,
            {"sid": student_id},
        )
        time_budget = None
        prefs: Dict[str, Any] = {}
        if srow:
            style = srow[0].get("style")
            if style:
                prefs["learning_style_preference"] = style
            try:
                tav = srow[0].get("time_availability")
                time_budget = int(float(tav)) if tav is not None else None
            except Exception:
                time_budget = None

        # Pull ordered learning data
        rows = execute_cypher_query(
            driver,
            """
            MATCH (:Student {StudentID: $sid})-[:HAS_LEARNING_DATA]->(ld:LearningData)
            WITH ld ORDER BY ld.timestamp ASC
            RETURN ld.node_id AS node_id, toFloat(ld.score) AS score, toInteger(ld.time_spent) AS time_spent
            """,
            {"sid": student_id},
        )

        # Group by node and compute mastery with recency decay
        by_node: Dict[str, list] = {}
        for r in rows:
            nid = str(r.get("node_id"))
            sc = r.get("score")
            ts = r.get("time_spent")
            if nid and sc is not None:
                by_node.setdefault(nid, []).append((float(sc), int(ts or 0)))

        mastery: Dict[str, float] = {}
        for nid, seq in by_node.items():
            # Normalize score to [0,1]; simple EMA with recency weight base FORGETTING_BASE
            n = len(seq)
            if n == 0:
                continue
            ema = 0.0
            denom = 0.0
            for idx, (score_pct, _t) in enumerate(seq):
                s01 = max(0.0, min(1.0, score_pct / 100.0))
                # more recent -> higher weight; position idx grows from 0..n-1
                recency = Config.FORGETTING_BASE ** (n - 1 - idx)
                ema += s01 * recency
                denom += recency
            mastery[nid] = ema / denom if denom > 0 else 0.0

        return cls(student_id=student_id, mastery=mastery, time_budget_minutes=time_budget, preferences=prefs)

    def predict_mastery(self, concept_id: str) -> Dict[str, Any]:
        """
        Baseline knowledge tracing: return current mastery and a next review date using
        the same scheduling rule. Acts as a simple, transparent default.
        """
        m = self.get_mastery(concept_id, 0.0)
        return {
            "concept_id": concept_id,
            "predicted_mastery": m,
            "next_review": self.estimate_next_review(concept_id),
            "confidence": 0.5 + 0.5 * m,  # naive mapping
        }
