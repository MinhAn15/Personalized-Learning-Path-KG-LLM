from __future__ import annotations

from typing import Any, Dict, List, Optional

from .config import Config


def explain_node_choice(node: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> str:
    """
    Provide a concise human-readable reason for choosing a node.
    This is a heuristic template; callers can enrich with real metrics.
    """
    title = node.get("title") or node.get("name") or node.get("id", "node")
    sim = node.get("similarity", 0.0)
    pri = node.get("priority", 0.0)
    diff = node.get("difficulty", 0.0)
    tmin = node.get("time_estimate", 0)

    pieces: List[str] = []
    if sim:
        pieces.append(f"high content match ({sim:.2f})")
    if pri:
        pieces.append(f"priority {pri:.1f}")
    if diff:
        pieces.append(f"difficulty {diff:.1f}")
    if tmin:
        pieces.append(f"~{int(round(tmin))} min")

    if not pieces:
        return f"Selected {title} based on graph proximity and prerequisites."
    return f"Selected {title} due to " + ", ".join(pieces) + "."


def explain_path(
    nodes: List[Dict[str, Any]],
    metrics: Optional[Dict[str, Any]] = None,
    learner: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Return an explainable summary for the chosen learning path.

    Output schema
    - summary: high-level description
    - steps: list of per-node rationales
    - metrics: passthrough of scoring metrics (cost, similarity, etc.)
    - caveats: known tradeoffs
    """
    steps = [explain_node_choice(n, metrics) for n in nodes]

    total_time = sum(int(n.get("time_estimate", 0) or 0) for n in nodes)
    avg_diff = (
        sum(float(n.get("difficulty", 0) or 0.0) for n in nodes) / max(1, len(nodes))
    )

    summary_bits = [
        f"Path length: {len(nodes)} nodes",
        f"Estimated time: ~{total_time} minutes",
        f"Average difficulty: {avg_diff:.2f}",
    ]
    if learner and getattr(learner, "time_budget_minutes", None):
        budget = learner.time_budget_minutes
        if budget is not None and total_time > budget:
            summary_bits.append(
                f"Note: exceeds time budget ({total_time}>{budget} min); consider pruning."
            )

    return {
        "summary": "; ".join(summary_bits),
        "steps": steps,
        "metrics": metrics or {},
        "caveats": [
            "Heuristics approximate fit; validate with learner feedback.",
            "Prerequisite gaps may require remediation steps.",
        ],
    }


def generate_counterfactuals(
    path_nodes: List[Dict[str, Any]],
    alternatives: List[Dict[str, Any]],
    k: int = 3,
) -> List[Dict[str, Any]]:
    """
    Suggest alternative nodes to swap into the path for exploration.
    This is a simple similarity+time heuristic placeholder.
    """
    if not alternatives:
        return []

    # Rank alternatives by (similarity desc, time asc)
    def alt_key(n: Dict[str, Any]):
        return (float(n.get("similarity", 0.0)), -float(n.get("time_estimate", 0.0)))

    ranked = sorted(alternatives, key=alt_key, reverse=True)
    picks = ranked[: max(0, k)]

    results = []
    for p in picks:
        results.append(
            {
                "node": p,
                "justification": explain_node_choice(p),
                "expected_impact": {
                    "time_delta_min": int(p.get("time_estimate", 0) or 0)
                },
            }
        )
    return results
