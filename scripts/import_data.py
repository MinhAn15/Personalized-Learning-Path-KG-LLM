#!/usr/bin/env python
"""Utility to load Knowledge Graph CSV files into Neo4j.

Supports two modes:
- auto/github: reuse backend.src.data_loader.check_and_load_kg (GitHub with local fallback)
- local: load explicit CSV paths passed via --nodes/--relationships
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List

import pandas as pd
from neo4j import GraphDatabase

from backend.src.config import Config, NEO4J_CONFIG
from backend.src.data_loader import check_and_load_kg, execute_cypher_query

CHUNK_SIZE = 200


def load_from_local(driver, nodes_path: Path, relationships_path: Path) -> Dict[str, int]:
    if not nodes_path.is_file():
        raise FileNotFoundError(f"Nodes file not found: {nodes_path}")
    if not relationships_path.is_file():
        raise FileNotFoundError(f"Relationships file not found: {relationships_path}")

    df_nodes = pd.read_csv(nodes_path, dtype=str).fillna("")
    with driver.session(database="neo4j") as session:
        for start in range(0, len(df_nodes), CHUNK_SIZE):
            chunk = df_nodes.iloc[start : start + CHUNK_SIZE]
            rows = chunk.to_dict(orient="records")
            cypher = (
                "UNWIND $rows AS row "
                f"MERGE (n:KnowledgeNode {{{Config.PROPERTY_ID}: row['{Config.PROPERTY_ID}']}}) "
                "SET n += row, "
                f"    n.{Config.PROPERTY_PRIORITY} = toInteger(row['{Config.PROPERTY_PRIORITY}']), "
                f"    n.{Config.PROPERTY_TIME_ESTIMATE} = toFloat(row['{Config.PROPERTY_TIME_ESTIMATE}'])"
            )
            session.run(cypher, rows=rows)

    df_rels = pd.read_csv(relationships_path, dtype=str).fillna("")
    with driver.session(database="neo4j") as session:
        for start in range(0, len(df_rels), CHUNK_SIZE):
            chunk = df_rels.iloc[start : start + CHUNK_SIZE]
            rows = chunk.to_dict(orient="records")
            tx = session.begin_transaction()
            for row in rows:
                rel_type = row.get("Relationship_Type") or row.get("RelationshipType") or "RELATED_TO"
                rel_type = "".join(c if c.isalnum() or c == "_" else "_" for c in rel_type).upper() or "RELATED_TO"
                source_id = row.get(Config.PROPERTY_SOURCE_ID) or row.get("source") or row.get("source_id")
                target_id = row.get(Config.PROPERTY_TARGET_ID) or row.get("target") or row.get("target_id")
                weight = row.get("Weight")
                dependency = row.get("Dependency")
                cypher = (
                    f"MATCH (s:KnowledgeNode {{{Config.PROPERTY_ID}: $src}}), "
                    f"      (t:KnowledgeNode {{{Config.PROPERTY_ID}: $tgt}}) "
                    f"MERGE (s)-[r:{rel_type}]->(t) "
                    "SET r.Weight = toFloat($weight), r.Dependency = toFloat($dependency)"
                )
                tx.run(cypher, src=source_id, tgt=target_id, weight=weight, dependency=dependency)
            tx.commit()

    return {
        "nodes_loaded": len(df_nodes),
        "relationships_loaded": len(df_rels),
    }


def summarize(driver) -> Dict[str, int]:
    node_count = execute_cypher_query(driver, "MATCH (n:KnowledgeNode) RETURN count(n) AS count")[0]["count"]
    rel_count = execute_cypher_query(driver, "MATCH ()-[r]->() RETURN count(r) AS count")[0]["count"]
    student_count = execute_cypher_query(driver, "MATCH (s:Student) RETURN count(s) AS count")[0]["count"]
    return {
        "knowledge_nodes": node_count,
        "relationships": rel_count,
        "students": student_count,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load KG CSV files into Neo4j")
    parser.add_argument(
        "--source",
        choices=["auto", "github", "local"],
        default="auto",
        help="Data source: auto/github uses check_and_load_kg; local loads provided CSV paths",
    )
    parser.add_argument("--nodes", type=Path, help="Path to nodes CSV (required for local mode)")
    parser.add_argument("--relationships", type=Path, help="Path to relationships CSV (required for local mode)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = NEO4J_CONFIG
    if not all(cfg.values()):
        raise RuntimeError("NEO4J configuration missing. Populate .env or set environment variables.")

    driver = GraphDatabase.driver(cfg["url"], auth=(cfg["username"], cfg["password"]))
    try:
        if args.source in ("auto", "github"):
            result = check_and_load_kg(driver)
            print(f"check_and_load_kg result: {result}")
            if args.source == "github":
                summary = summarize(driver)
                print(f"Graph summary: {summary}")
                return
        if args.source == "local" or (args.source == "auto" and result.get("status") != "success"):
            if not args.nodes or not args.relationships:
                raise ValueError("--nodes and --relationships are required for local mode")
            stats = load_from_local(driver, args.nodes, args.relationships)
            print(f"Loaded local CSVs: {stats}")
            summary = summarize(driver)
            print(f"Graph summary: {summary}")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
