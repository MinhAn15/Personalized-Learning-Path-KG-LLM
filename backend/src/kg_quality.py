from __future__ import annotations

from typing import Any, Dict, List, Tuple


def detect_inconsistencies(neo4j_driver: Any) -> List[Dict[str, str]]:
    """
    Run a set of lightweight checks to flag potential KG inconsistencies.
    Returns a list of {type, detail} records.
    """
    issues: List[Dict[str, str]] = []
    if neo4j_driver is None:
        issues.append({"type": "connection", "detail": "Neo4j driver not provided"})
        return issues

    with neo4j_driver.session() as sess:
        # Example: duplicate node titles
        q1 = """
        MATCH (n:KnowledgeNode)
        WITH n.title AS t, count(*) AS c
        WHERE t IS NOT NULL AND c > 1
        RETURN t AS title, c AS dup_count
        LIMIT 50
        """
        for rec in sess.run(q1):
            issues.append(
                {
                    "type": "duplicate_title",
                    "detail": f"title={rec['title']} duplicates={rec['dup_count']}",
                }
            )

        # Example: negative or missing time estimates
        q2 = """
        MATCH (n:KnowledgeNode)
        WHERE n.Time_Estimate IS NULL OR n.Time_Estimate < 0
        RETURN n.title AS title
        LIMIT 50
        """
        for rec in sess.run(q2):
            issues.append({"type": "time_estimate", "detail": f"{rec['title']}"})

    return issues


def check_prereq_cycles(neo4j_driver: Any) -> List[List[str]]:
    """
    Detect cycles in prerequisite relations. Returns samples of cyclic paths.
    """
    samples: List[List[str]] = []
    if neo4j_driver is None:
        return samples

    with neo4j_driver.session() as sess:
        q = """
        MATCH p=(a:KnowledgeNode)-[:PREREQUISITE_OF*1..6]->(a)
        RETURN [n IN nodes(p) | n.title][0..10] AS cycle
        LIMIT 10
        """
        for rec in sess.run(q):
            samples.append([t for t in rec["cycle"] if t is not None])
    return samples


def validate_difficulty_ranges(neo4j_driver: Any) -> Dict[str, float]:
    """
    Compute simple stats to validate difficulty coverage and outliers.
    """
    if neo4j_driver is None:
        return {"count": 0.0}

    with neo4j_driver.session() as sess:
        q = """
        MATCH (n:KnowledgeNode)
        WITH n, coalesce(n.Difficulty, 0.0) AS d
        RETURN count(n) AS cnt, avg(d) AS avg, stdev(d) AS sd, min(d) AS min, max(d) AS max
        """
        rec = sess.run(q).single()
        return {
            "count": float(rec["cnt"] or 0.0),
            "avg": float(rec["avg"] or 0.0),
            "stdev": float(rec["sd"] or 0.0),
            "min": float(rec["min"] or 0.0),
            "max": float(rec["max"] or 0.0),
        }


def compute_node_coverage(
    neo4j_driver: Any, labels: Tuple[str, ...] = ("KnowledgeNode",)
) -> float:
    """
    Percentage of nodes that have both difficulty and time estimates.
    """
    if neo4j_driver is None:
        return 0.0

    with neo4j_driver.session() as sess:
        q = f"""
        MATCH (n:{':'.join(labels)})
        WITH n,
             exists(n.Difficulty) AS hasD,
             exists(n.Time_Estimate) AS hasT
        RETURN toFloat(sum(CASE WHEN hasD AND hasT THEN 1 ELSE 0 END))/toFloat(count(n)) AS coverage
        """
        rec = sess.run(q).single()
        return float(rec["coverage"] or 0.0)
