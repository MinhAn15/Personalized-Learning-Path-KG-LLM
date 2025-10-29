from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .config import Config
from .learner_state import LearnerState

try:
    # Optional reuse of existing A* if available
    from .path_generator import (
        a_star_custom,
        determine_start_node,
        determine_goal_node,
        calculate_dynamic_weights,
    )
except Exception:  # pragma: no cover - optional integration
    a_star_custom = None  # type: ignore
    determine_start_node = None  # type: ignore
    determine_goal_node = None  # type: ignore
    calculate_dynamic_weights = None  # type: ignore


class AdaptivePathPlanner:
    """
    High-level adapter for personalized path planning.

    This wrapper optionally delegates to the existing A* if present, while
    allowing injection of learner-centric dynamic weights and constraints.
    The goal is to keep this non-breaking and easily swappable.
    """

    def __init__(self, neo4j_driver: Any | None = None) -> None:
        self.driver = neo4j_driver

    # --- Dynamic weight logic -------------------------------------------------
    def compute_dynamic_weights(self, learner: LearnerState) -> Dict[str, float]:
        """
        Convert LearnerState into weight scalars compatible with heuristic terms.
        This mirrors calculate_dynamic_weights but allows additional signals.
        """
        # Time scalar: slower learners value time penalty more; if user set pace
        pace = learner.preferences.get("pace", "normal")
        base_time = 1.0
        if pace == "fast":
            base_time = 0.75
        elif pace == "slow":
            base_time = 1.25

        # Optional mastery-driven emphasis: if low mastery overall, reduce time weight
        if learner.mastery:
            avg_mastery = sum(learner.mastery.values()) / max(1, len(learner.mastery))
        else:
            avg_mastery = 0.0
        mastery_factor = 0.9 + (1.0 - avg_mastery) * 0.2  # 0.9..1.1

        dynamic = {
            "difficulty_standard": 1.0,
            "difficulty_advanced": 1.0,
            "skill_level": 1.0,
            "time_estimate": base_time * mastery_factor,
        }
        return dynamic

    # --- Planning API ---------------------------------------------------------
    def plan_path(
        self,
        learner: LearnerState,
        start_concept: Optional[str] = None,
        goal_concept: Optional[str] = None,
        max_depth: int = 50,
        hard_time_budget_minutes: Optional[int] = None,
        extra_constraints: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Produce a personalized path plan. Delegates to a_star_custom when available.

        Returns a dict including: nodes, total_cost, meta.
        """
        if a_star_custom is None:
            # Skeleton fallback when A* is unavailable
            return {
                "nodes": [],
                "total_cost": 0.0,
                "meta": {
                    "note": "A* implementation not available; returning empty plan.",
                },
            }

        # Determine dynamic weights; optionally combine with existing function
        dyn = self.compute_dynamic_weights(learner)
        if calculate_dynamic_weights is not None:
            try:
                existing_dyn = calculate_dynamic_weights(learner.student_id)
                # Merge with precedence to our computed values when set
                dyn = {**existing_dyn, **dyn}
            except Exception:
                pass

        # Determine start/goal when not provided, if helpers exist
        if start_concept is None and determine_start_node is not None:
            try:
                start_concept = determine_start_node(learner.student_id)
            except Exception:
                start_concept = None
        if goal_concept is None and determine_goal_node is not None:
            try:
                goal_concept = determine_goal_node(learner.student_id)
            except Exception:
                goal_concept = None

        # Guard — cannot proceed without nodes
        if not start_concept or not goal_concept:
            return {
                "nodes": [],
                "total_cost": 0.0,
                "meta": {
                    "note": "Start or goal concept missing; returning empty plan.",
                },
            }

        # Delegate to existing A*
        try:
            result = a_star_custom(
                start_concept,
                goal_concept,
                dynamic_weights=dyn,
                max_depth=max_depth,
                time_budget_minutes=hard_time_budget_minutes,
                extra_constraints=extra_constraints or {},
            )
        except TypeError:
            # Older signature that doesn't support new params
            result = a_star_custom(
                start_concept,
                goal_concept,
                dynamic_weights=dyn,
            )

        return {
            "nodes": result.get("path", []),
            "total_cost": result.get("total_cost", 0.0),
            "meta": {
                "expanded": result.get("expanded", 0),
                "visited": result.get("visited", 0),
                "weights": dyn,
            },
        }
