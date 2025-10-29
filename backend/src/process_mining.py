from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from datetime import datetime
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Optional

from .data_loader import execute_cypher_query

logger = logging.getLogger(__name__)

try:
    from pm4py.algo.conformance.alignments import factory as alignments_factory  # type: ignore
    from pm4py.algo.discovery.inductive import factory as inductive_miner_factory  # type: ignore
    from pm4py.algo.discovery.log_skeleton.algorithm import apply as log_skeleton_apply  # type: ignore
    from pm4py.objects.log.obj import EventLog, Event, Trace  # type: ignore
    from pm4py.util.xes_constants import DEFAULT_CASE_CONCEPT_NAME, DEFAULT_NAME_KEY, DEFAULT_TIMESTAMP_KEY  # type: ignore
    PM4PY_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    EventLog = Trace = Event = None  # type: ignore
    DEFAULT_CASE_CONCEPT_NAME = "case:concept:name"
    DEFAULT_NAME_KEY = "concept:name"
    DEFAULT_TIMESTAMP_KEY = "time:timestamp"
    alignments_factory = None  # type: ignore
    inductive_miner_factory = None  # type: ignore
    log_skeleton_apply = None  # type: ignore
    PM4PY_AVAILABLE = False


@dataclass
class LearningEvent:
    student_id: str
    node_id: str
    timestamp: datetime
    recommended_index: Optional[int] = None
    position_in_path: Optional[int] = None
    duration_minutes: Optional[float] = None

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> Optional["LearningEvent"]:
        try:
            student_id = str(row.get("student_id") or row.get("StudentID") or row.get("student"))
            node_id = str(row.get("node_id") or row.get("Node_ID") or row.get("concept"))
            if not student_id or not node_id:
                return None
            ts_raw = row.get("timestamp") or row.get("Timestamp") or row.get("time")
            if ts_raw is None:
                return None
            timestamp = _parse_timestamp(ts_raw)
            rec_index = row.get("recommended_index") or row.get("recommended_order")
            pos_index = row.get("step_index") or row.get("position")
            duration_raw = row.get("time_spent") or row.get("duration_minutes")
            return cls(
                student_id=student_id,
                node_id=node_id,
                timestamp=timestamp,
                recommended_index=_safe_int(rec_index),
                position_in_path=_safe_int(pos_index),
                duration_minutes=_safe_float(duration_raw),
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Skipping row during LearningEvent parsing: %s", exc)
            return None


def load_learning_events_from_neo4j(
    driver: Any,
    student_ids: Optional[Iterable[str]] = None,
    limit: Optional[int] = None,
) -> List[LearningEvent]:
    query = """
    MATCH (s:Student)-[:HAS_LEARNING_DATA]->(ld:LearningData)
    WHERE $student_ids IS NULL OR s.StudentID IN $student_ids
    RETURN s.StudentID AS student_id,
           ld.node_id AS node_id,
           coalesce(ld.timestamp, ld.updated_at, ld.created_at) AS timestamp,
           ld.recommended_index AS recommended_index,
           ld.position AS position,
           ld.time_spent AS time_spent
    ORDER BY student_id, timestamp
    LIMIT $limit
    """
    params = {
        "student_ids": list(student_ids) if student_ids is not None else None,
        "limit": int(limit) if limit is not None else 10_000,
    }
    rows = execute_cypher_query(driver, query, params)
    events: List[LearningEvent] = []
    for row in rows:
        mapped = {
            "student_id": row.get("student_id"),
            "node_id": row.get("node_id"),
            "timestamp": row.get("timestamp"),
            "recommended_index": row.get("recommended_index"),
            "position": row.get("position"),
            "time_spent": row.get("time_spent"),
        }
        event = LearningEvent.from_row(mapped)
        if event:
            events.append(event)
    logger.info("Loaded %d learning events from Neo4j", len(events))
    return events


def load_learning_events_from_csv(path: str) -> List[LearningEvent]:
    events: List[LearningEvent] = []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            event = LearningEvent.from_row(row)
            if event:
                events.append(event)
    logger.info("Loaded %d learning events from %s", len(events), path)
    return events


def load_recommended_paths_from_csv(path: str) -> Dict[str, List[str]]:
    paths: Dict[str, List[str]] = {}
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            student_id = str(row.get("student_id") or row.get("StudentID") or "").strip()
            node_id = str(row.get("node_id") or row.get("Node_ID") or row.get("concept") or "").strip()
            step_raw = row.get("step_index") or row.get("order") or row.get("position")
            if not student_id or not node_id:
                continue
            paths.setdefault(student_id, []).append((node_id, _safe_int(step_raw)))
    normalized: Dict[str, List[str]] = {}
    for student_id, nodes in paths.items():
        nodes.sort(key=lambda item: item[1] if item[1] is not None else 10_000)
        normalized[student_id] = [node for node, _ in nodes]
    return normalized


def build_event_log(
    events: Iterable[LearningEvent],
    case_id_key: str = "student_id",
    activity_key: str = "node_id",
    timestamp_key: str = "timestamp",
):
    if not PM4PY_AVAILABLE:
        logger.warning("pm4py not available; returning raw trace list.")
        grouped = _group_events_by_case(events)
        return grouped

    log = EventLog()
    grouped = _group_events_by_case(events)
    for student_id, student_events in grouped.items():
        trace = Trace()
        trace.attributes[DEFAULT_CASE_CONCEPT_NAME] = student_id
        for event in student_events:
            evt = Event({
                DEFAULT_NAME_KEY: getattr(event, activity_key),
                DEFAULT_TIMESTAMP_KEY: getattr(event, timestamp_key),
                case_id_key: student_id,
            })
            if event.duration_minutes is not None:
                evt["duration_minutes"] = event.duration_minutes
            if event.recommended_index is not None:
                evt["recommended_index"] = event.recommended_index
            if event.position_in_path is not None:
                evt["position_in_path"] = event.position_in_path
            trace.append(evt)
        log.append(trace)
    return log


def build_log_skeleton(event_log):
    if not PM4PY_AVAILABLE or log_skeleton_apply is None or not isinstance(event_log, EventLog):
        logger.warning("pm4py log skeleton unavailable; returning None.")
        return None
    try:
        skeleton = log_skeleton_apply(event_log)
        return skeleton
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to build log skeleton: %s", exc)
        return None


def discover_process_model(event_log):
    if not PM4PY_AVAILABLE or inductive_miner_factory is None or not isinstance(event_log, EventLog):
        logger.warning("pm4py inductive miner unavailable; returning None.")
        return None
    try:
        net, initial_marking, final_marking = inductive_miner_factory.apply(event_log)
        return {
            "net": net,
            "initial_marking": initial_marking,
            "final_marking": final_marking,
        }
    except Exception as exc:  # pragma: no cover
        logger.warning("Process discovery failed: %s", exc)
        return None


def analyze_learning_process(
    events: Iterable[LearningEvent],
    recommended_paths: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Any]:
    events_list = sorted(list(events), key=lambda e: (e.student_id, e.timestamp))
    event_log = build_event_log(events_list)
    skeleton = build_log_skeleton(event_log)
    model = discover_process_model(event_log)

    metrics = compute_metrics(events_list, recommended_paths or {}, event_log=event_log, model=model)
    return {
        "event_log": event_log,
        "log_skeleton": skeleton,
        "process_model": model,
        "metrics": metrics,
    }


def compute_metrics(
    events: List[LearningEvent],
    recommended_paths: Dict[str, List[str]],
    event_log: Any = None,
    model: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    actual_paths = _group_events_by_case(events)
    path_sequences: Dict[str, List[str]] = {}
    completion_times: List[float] = []
    per_student: Dict[str, Dict[str, Any]] = {}

    for student_id, student_events in actual_paths.items():
        ordered = sorted(student_events, key=lambda e: e.timestamp)
        sequence = [event.node_id for event in ordered]
        path_sequences[student_id] = sequence
        duration = _compute_completion_minutes(ordered)
        if duration is not None:
            completion_times.append(duration)
        recommended = recommended_paths.get(student_id, [])
        conformance = _lcs_ratio(sequence, recommended)
        deviation = _average_position_deviation(sequence, recommended)
        per_student[student_id] = {
            "actual_sequence": sequence,
            "recommended_sequence": recommended,
            "conformance": conformance,
            "average_deviation": deviation,
            "completion_minutes": duration,
        }

    conformance_values = [entry["conformance"] for entry in per_student.values() if entry["conformance"] is not None]
    deviation_values = [entry["average_deviation"] for entry in per_student.values() if entry["average_deviation"] is not None]

    metrics = {
        "conformance_rate": mean(conformance_values) if conformance_values else None,
        "average_deviation": mean(deviation_values) if deviation_values else None,
        "completion_time_variance": pstdev(completion_times) ** 2 if len(completion_times) > 1 else 0.0,
        "per_student": per_student,
    }

    if PM4PY_AVAILABLE and isinstance(event_log, EventLog) and model is not None:  # pragma: no cover
        try:
            net = model.get("net")
            initial_marking = model.get("initial_marking")
            final_marking = model.get("final_marking")
            if net and alignments_factory is not None:
                alignments = alignments_factory.apply_log(event_log, net, initial_marking, final_marking)
                conformance_scores = [1.0 - (alignment.get("cost", 0.0) / max(1.0, alignment.get("fitness", 1.0))) for alignment in alignments]
                metrics["alignment_conformance"] = mean(conformance_scores)
        except Exception as exc:
            logger.debug("Alignment-based conformance failed: %s", exc)

    return metrics


def _group_events_by_case(events: Iterable[LearningEvent]) -> Dict[str, List[LearningEvent]]:
    grouped: Dict[str, List[LearningEvent]] = {}
    for event in events:
        grouped.setdefault(event.student_id, []).append(event)
    return grouped


def _compute_completion_minutes(events: List[LearningEvent]) -> Optional[float]:
    if not events:
        return None
    start = events[0].timestamp
    end = events[-1].timestamp
    delta = end - start
    return delta.total_seconds() / 60.0


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None or value == "":
            return None
        return int(float(value))
    except Exception:
        return None


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _parse_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value))
    value_str = str(value).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f", "%d/%m/%Y %H:%M"):
        try:
            return datetime.strptime(value_str, fmt)
        except ValueError:
            continue
    return datetime.fromisoformat(value_str)


def _lcs_ratio(actual: List[str], recommended: List[str]) -> Optional[float]:
    if not actual or not recommended:
        return None
    m, n = len(actual), len(recommended)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            if actual[i] == recommended[j]:
                dp[i][j] = 1 + dp[i + 1][j + 1]
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])
    lcs = dp[0][0]
    return lcs / max(len(actual), len(recommended))


def _average_position_deviation(actual: List[str], recommended: List[str]) -> Optional[float]:
    if not actual or not recommended:
        return None
    index_map = {node: idx for idx, node in enumerate(recommended)}
    diffs: List[float] = []
    for idx, node in enumerate(actual):
        if node in index_map:
            diffs.append(abs(idx - index_map[node]))
    if not diffs:
        return float(max(len(actual), len(recommended)))
    return float(mean(diffs))

```}