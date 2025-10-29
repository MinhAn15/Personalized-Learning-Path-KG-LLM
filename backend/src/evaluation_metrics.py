from __future__ import annotations

import csv
import logging
import math
from collections import Counter
from dataclasses import dataclass
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence

logger = logging.getLogger(__name__)


@dataclass
class MetricResult:
    name: str
    value: float
    details: Dict[str, Any]


def _safe_len(sequence: Optional[Sequence[Any]]) -> int:
    return len(sequence) if sequence is not None else 0


def accuracy_correct_order(
    recommended: Sequence[Any],
    ideal_order: Sequence[Any],
    key: str = "node_id",
) -> MetricResult:
    ideal_lookup = {item if isinstance(item, str) else item.get(key): idx for idx, item in enumerate(ideal_order)}
    correct_positions = 0
    total = 0
    for idx, item in enumerate(recommended):
        node_id = item if isinstance(item, str) else item.get(key)
        if node_id is None:
            continue
        total += 1
        if ideal_lookup.get(node_id) == idx:
            correct_positions += 1
    accuracy = float(correct_positions) / float(total) if total else 0.0
    return MetricResult(
        name="accuracy_order",
        value=accuracy,
        details={
            "correct_positions": correct_positions,
            "total_positions": total,
            "ideal_count": len(ideal_lookup),
        },
    )


def diversity_concept_variety(
    recommended: Sequence[Any],
    concept_key: str = "concept_type",
) -> MetricResult:
    concept_count = Counter()
    for item in recommended:
        if isinstance(item, dict):
            concept = item.get(concept_key) or item.get("category") or item.get("skill_level")
        else:
            concept = str(item)
        if concept:
            concept_count[str(concept).lower()] += 1
    total = sum(concept_count.values())
    distinct = len(concept_count)
    entropy = 0.0
    if total:
        entropy = -sum((count / total) * math.log(count / total, 2) for count in concept_count.values())
    normalized_variety = distinct / total if total else 0.0
    diversity_score = min(1.0, (normalized_variety + entropy / math.log(max(distinct, 2), 2)) / 2.0) if distinct > 1 else normalized_variety
    return MetricResult(
        name="diversity",
        value=float(diversity_score),
        details={
            "distinct": distinct,
            "total": total,
            "concept_counts": dict(concept_count),
            "normalized_variety": normalized_variety,
            "entropy": entropy,
        },
    )


def path_length_efficiency(
    recommended: Sequence[Any],
    baseline_length: Optional[int] = None,
) -> MetricResult:
    length = _safe_len(recommended)
    if baseline_length is None:
        efficiency = 1.0 if length else 0.0
    else:
        if baseline_length == 0:
            efficiency = 1.0 if length == 0 else 0.0
        else:
            efficiency = float(baseline_length) / float(length) if length else 0.0
    efficiency = float(min(max(efficiency, 0.0), 1.5))
    return MetricResult(
        name="path_length_efficiency",
        value=efficiency,
        details={
            "recommended_length": length,
            "baseline_length": baseline_length,
        },
    )


def student_satisfaction_placeholder(
    survey_scores: Optional[Iterable[float]] = None,
    default_value: float = 0.75,
) -> MetricResult:
    scores: List[float] = []
    if survey_scores:
        for score in survey_scores:
            try:
                scores.append(float(score))
            except (TypeError, ValueError):
                continue
    value = mean(scores) if scores else default_value
    value = float(min(max(value, 0.0), 1.0))
    return MetricResult(
        name="student_satisfaction",
        value=value,
        details={
            "provided_scores": len(scores),
            "default_used": not scores,
        },
    )


def evaluate_learning_path(
    recommended: Sequence[Any],
    *,
    ideal_order: Optional[Sequence[Any]] = None,
    baseline_length: Optional[int] = None,
    survey_scores: Optional[Iterable[float]] = None,
) -> Dict[str, MetricResult]:
    metrics = {}
    if ideal_order is not None:
        metrics["accuracy_order"] = accuracy_correct_order(recommended, ideal_order)
    metrics["diversity"] = diversity_concept_variety(recommended)
    metrics["path_length_efficiency"] = path_length_efficiency(recommended, baseline_length)
    metrics["student_satisfaction"] = student_satisfaction_placeholder(survey_scores)
    return metrics


def export_metrics_to_csv(metrics: Dict[str, MetricResult], filepath: str) -> None:
    fieldnames = ["metric", "value"]
    with open(filepath, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for name, result in metrics.items():
            writer.writerow({"metric": name, "value": f"{result.value:.4f}"})


def metrics_summary(metrics: Dict[str, MetricResult]) -> Dict[str, Any]:
    summary = {name: result.value for name, result in metrics.items()}
    summary["details"] = {name: result.details for name, result in metrics.items()}
    return summary
*** End of File
