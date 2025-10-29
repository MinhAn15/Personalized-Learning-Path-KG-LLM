"""
Neo4j schema setup: constraints and indexes for performance and integrity.

Run against a dev/test database. Uses Config.NEO4J_CONFIG.

Usage (PowerShell):
  python backend/scripts/neo4j_schema_setup.py
"""

import os
from neo4j import GraphDatabase
from backend.src.config import Config

DDL_STATEMENTS = [
    # KnowledgeNode unique id (existing model)
    ("constraint", "CREATE CONSTRAINT knowledge_node_id IF NOT EXISTS FOR (n:KnowledgeNode) REQUIRE n.%s IS UNIQUE" % Config.PROPERTY_ID),
    # Student unique id (existing 'Student' label if used)
    ("constraint", "CREATE CONSTRAINT student_id IF NOT EXISTS FOR (s:Student) REQUIRE s.StudentID IS UNIQUE"),
    # Useful BTREE indexes for query filters
    ("index", f"CREATE INDEX knowledge_node_context IF NOT EXISTS FOR (n:KnowledgeNode) ON (n.{Config.PROPERTY_CONTEXT})"),
    ("index", f"CREATE INDEX knowledge_node_skill IF NOT EXISTS FOR (n:KnowledgeNode) ON (n.{Config.PROPERTY_SKILL_LEVEL})"),
    ("index", f"CREATE INDEX knowledge_node_priority IF NOT EXISTS FOR (n:KnowledgeNode) ON (n.{Config.PROPERTY_PRIORITY})"),
    ("index", f"CREATE INDEX knowledge_node_difficulty IF NOT EXISTS FOR (n:KnowledgeNode) ON (n.{Config.PROPERTY_DIFFICULTY})"),
    ("index", f"CREATE INDEX knowledge_node_time IF NOT EXISTS FOR (n:KnowledgeNode) ON (n.{Config.PROPERTY_TIME_ESTIMATE})"),
    ("index", f"CREATE INDEX knowledge_node_tags IF NOT EXISTS FOR (n:KnowledgeNode) ON (n.{Config.PROPERTY_SEMANTIC_TAGS})"),
    # Student frequently filtered fields
    ("index", "CREATE INDEX student_level IF NOT EXISTS FOR (s:Student) ON (s.current_level)"),
    ("index", "CREATE INDEX student_style IF NOT EXISTS FOR (s:Student) ON (s.learning_style_preference)"),
    # LearningData composite index (student_id,timestamp) for time series queries
    ("index", "CREATE INDEX learning_data_student_time IF NOT EXISTS FOR (ld:LearningData) ON (ld.student_id, ld.timestamp)"),
    ("index", "CREATE INDEX learning_data_node IF NOT EXISTS FOR (ld:LearningData) ON (ld.node_id)"),
    # Relationship property index (Neo4j 5+). Ignored if unsupported.
    ("rel_index", "CREATE INDEX rel_weight IF NOT EXISTS FOR ()-[r]->() ON (r.Weight)"),
    ("rel_index", "CREATE INDEX rel_dependency IF NOT EXISTS FOR ()-[r]->() ON (r.Dependency)"),
]


def run():
    cfg = Config.NEO4J_CONFIG
    if not all(cfg.values()):
        raise RuntimeError("NEO4J configuration missing. Set NEO4J_URL, NEO4J_USER, NEO4J_PASSWORD in .env")
    driver = GraphDatabase.driver(cfg["url"], auth=(cfg["username"], cfg["password"]))
    try:
        with driver.session(database="neo4j") as session:
            for kind, stmt in DDL_STATEMENTS:
                try:
                    session.run(stmt)
                    print(f"Applied {kind}: {stmt}")
                except Exception as e:
                    # Non-fatal: print and continue (e.g., older Neo4j lacking rel prop index)
                    print(f"Skip {kind}: {e}")
    finally:
        driver.close()


if __name__ == "__main__":
    run()
