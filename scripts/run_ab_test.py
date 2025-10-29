#!/usr/bin/env python
"""Run an A/B test comparing A*, RL, and Hybrid path planners.

The script pulls learner profiles from Neo4j, plans learning paths with three
strategies, evaluates them using common metrics, and writes a CSV summary plus
an optional bar chart of average performance.

Usage:
    python scripts/run_ab_test.py --learners student-1 student-2
    python scripts/run_ab_test.py --limit 5 --output ./ab_results.csv

Outputs:
    - CSV table of metrics per learner/planner (default: backend/data/output/ab_test_results.csv)
    - PNG bar chart of averaged metrics (default: backend/data/output/ab_test_metrics.png)
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Lazy imports of project modules
from backend.src.config import Config
from backend.src.adaptive_path_planner import AdaptivePathPlanner
from backend.src.learner_state import LearnerState
from backend.src.evaluation_metrics import evaluate_learning_path, metrics_summary

try:  # Optional analytics dependencies
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore

try:  # Optional plotting dependency
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None  # type: ignore


@dataclass
class PlannerOutput:
    learner_id: str
    planner: str
    nodes: List[Dict[str, Any]]
    total_cost: Optional[float]
    meta: Dict[str, Any]


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def ensure_driver() -> Any:
    driver = getattr(Config, "NEO4J_DRIVER", None)
    if driver:
        return driver
    try:
        from backend.src.api import driver as api_driver  # pylint: disable=import-outside-toplevel

        if api_driver:
            return api_driver
    except Exception:  # pragma: no cover - ignore import issues
        pass
    try:
        from backend.src.main import initialize_connections_and_settings  # pylint: disable=import-outside-toplevel

        driver = initialize_connections_and_settings()
        if driver:
            Config.NEO4J_DRIVER = driver
        return driver
    except Exception as exc:  # pragma: no cover
        logging.error("Unable to initialize Neo4j driver: %s", exc)
        return None


def fetch_student_ids(driver: Any, limit: int) -> List[str]:
    if not driver:
        return []
    try:
        from backend.src.data_loader import execute_cypher_query  # pylint: disable=import-outside-toplevel

        rows = execute_cypher_query(
            driver,
            """
            MATCH (s:Student)
            RETURN s.StudentID AS sid
            ORDER BY sid ASC
            LIMIT $limit
            """,
            {"limit": limit},
        )
        return [str(row.get("sid")) for row in rows or [] if row.get("sid")]
    except Exception as exc:  # pragma: no cover
        logging.warning("Failed to fetch student IDs: %s", exc)
        return []


def normalise_nodes(nodes: Sequence[Any]) -> List[Dict[str, Any]]:
    normalised: List[Dict[str, Any]] = []
    for node in nodes:
        if isinstance(node, dict):
            normalised.append(dict(node))
        else:
            normalised.append({"node_id": str(node)})
    return normalised


def clone_learner_state(state: LearnerState) -> LearnerState:
    return LearnerState(
        student_id=state.student_id,
        mastery=dict(state.mastery),
        cognitive_level_goal=dict(state.cognitive_level_goal),
        time_budget_minutes=state.time_budget_minutes,
        preferences=dict(state.preferences),
    )


def collect_plans(planner: AdaptivePathPlanner, learner_state: LearnerState, learner_id: str) -> Dict[str, PlannerOutput]:
    plans: Dict[str, PlannerOutput] = {}

    # Baseline A*
    try:
        a_star_state = clone_learner_state(learner_state)
        a_star_result = planner.plan_path(a_star_state, mode="a_star")
        nodes = normalise_nodes(a_star_result.get("nodes", []) or [])
        plans["a_star"] = PlannerOutput(
            learner_id=learner_id,
            planner="A*",
            nodes=nodes,
            total_cost=a_star_result.get("total_cost"),
            meta=a_star_result.get("meta", {}),
        )
    except Exception as exc:
        logging.error("A* planning failed for %s: %s", learner_id, exc)

    # RL planner
    try:
        rl_state = clone_learner_state(learner_state)
        rl_result = planner.plan_path(rl_state, mode="rl")
        nodes = normalise_nodes(rl_result.get("nodes", []) or [])
        plans["rl"] = PlannerOutput(
            learner_id=learner_id,
            planner="RL",
            nodes=nodes,
            total_cost=rl_result.get("total_cost"),
            meta=rl_result.get("meta", {}),
        )
    except ImportError as exc:
        logging.warning("RL planning unavailable for %s: %s", learner_id, exc)
    except Exception as exc:
        logging.error("RL planning failed for %s: %s", learner_id, exc)

    # Hybrid planner (requires driver)
    try:
        hybrid_result = planner.plan_hybrid_path(learner_id)
        nodes = normalise_nodes(hybrid_result.get("nodes", []) or [])
        plans["hybrid"] = PlannerOutput(
            learner_id=learner_id,
            planner="Hybrid",
            nodes=nodes,
            total_cost=hybrid_result.get("total_cost"),
            meta=hybrid_result.get("meta", {}),
        )
    except Exception as exc:
        logging.error("Hybrid planning failed for %s: %s", learner_id, exc)

    return plans


def evaluate_plans(plans: Dict[str, PlannerOutput]) -> Dict[str, Dict[str, float]]:
    if not plans:
        return {}
    baseline = plans.get("a_star")
    ideal = baseline.nodes if baseline else []
    baseline_len = len(ideal) or None

    scores: Dict[str, Dict[str, float]] = {}
    for key, plan in plans.items():
        metrics = evaluate_learning_path(
            plan.nodes,
            ideal_order=ideal,
            baseline_length=baseline_len,
            survey_scores=None,
        )
        summary = metrics_summary(metrics)
        scores[key] = summary
    return scores


def flatten_results(
    learner_id: str,
    planner_key: str,
    plan: PlannerOutput,
    scores: Dict[str, float],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    details = scores.get("details", {}) if isinstance(scores, dict) else {}
    for metric_name, value in scores.items():
        if metric_name == "details":
            continue
        rows.append(
            {
                "learner_id": learner_id,
                "planner": plan.planner,
                "planner_key": planner_key,
                "metric": metric_name,
                "value": value,
                "nodes": len(plan.nodes),
                "total_cost": plan.total_cost,
                "meta": json.dumps(plan.meta, ensure_ascii=True),
                "details": json.dumps(details.get(metric_name, {}), ensure_ascii=True),
            }
        )
    return rows


def save_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
    if not rows:
        logging.warning("No results to save; skipping CSV export.")
        return
    fieldnames = list(rows[0].keys())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logging.info("Saved results to %s", output_path)


def plot_metrics(rows: List[Dict[str, Any]], output_path: Path) -> None:
    if not rows:
        logging.warning("No data available for plotting; skipping chart generation.")
        return
    if pd is None or plt is None:
        logging.warning("pandas/matplotlib not available; cannot generate chart.")
        return
    df = pd.DataFrame(rows)
    pivot = (
        df.groupby(["planner", "metric"], as_index=False)["value"]
        .mean()
        .pivot(index="metric", columns="planner", values="value")
    )
    pivot = pivot.sort_index()
    ax = pivot.plot(kind="bar", figsize=(8, 5))
    ax.set_title("Average Metric Scores by Planner")
    ax.set_ylabel("Score")
    ax.set_xlabel("Metric")
    ax.set_ylim(0, 1.2)
    ax.legend(title="Planner")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    logging.info("Saved chart to %s", output_path)


def run_experiments(learner_ids: Sequence[str], output_csv: Path, output_chart: Path) -> None:
    driver = ensure_driver()
    if not driver:
        logging.error("Neo4j driver unavailable; cannot run A/B test.")
        return

    planner = AdaptivePathPlanner(driver)
    rows: List[Dict[str, Any]] = []

    for learner_id in learner_ids:
        try:
            learner_state = LearnerState.from_neo4j(driver, learner_id)
        except Exception as exc:
            logging.error("Failed to build learner state for %s: %s", learner_id, exc)
            continue

        plans = collect_plans(planner, learner_state, learner_id)
        if not plans:
            logging.warning("No plans produced for %s; skipping.", learner_id)
            continue

        scores = evaluate_plans(plans)
        for key, plan in plans.items():
            score_summary = scores.get(key, {})
            rows.extend(flatten_results(learner_id, key, plan, score_summary))

    save_csv(rows, output_csv)
    plot_metrics(rows, output_chart)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare path planners via metrics.")
    parser.add_argument("--learners", nargs="*", help="Specific learner IDs to evaluate.")
    parser.add_argument("--limit", type=int, default=3, help="Number of learners to sample when --learners not provided.")
    parser.add_argument("--output", default=os.path.join("backend", "data", "output", "ab_test_results.csv"), help="Output CSV path.")
    parser.add_argument("--chart", default=os.path.join("backend", "data", "output", "ab_test_metrics.png"), help="Output PNG chart path.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    setup_logging(args.verbose)

    if args.learners:
        learner_ids = args.learners
    else:
        driver = ensure_driver()
        learner_ids = fetch_student_ids(driver, args.limit)
        if not learner_ids:
            logging.error("No learner IDs available; provide --learners explicitly.")
            return

    logging.info("Running A/B test for learners: %s", ", ".join(learner_ids))
    run_experiments(learner_ids, Path(args.output), Path(args.chart))


if __name__ == "__main__":
    main()
