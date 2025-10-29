from __future__ import annotations

import logging
from typing import Any, Dict, List

from neo4j import GraphDatabase, ManagedTransaction

from .config import NEO4J_CONFIG, Config

logger = logging.getLogger(__name__)


class Neo4jManager:
    """
    Connection/transaction manager for Neo4j with a convenient API.
    Uses environment configuration from Config.NEO4J_CONFIG.
    """

    def __init__(self, uri: str | None = None, username: str | None = None, password: str | None = None):
        cfg = NEO4J_CONFIG
        self.driver = GraphDatabase.driver(
            uri or cfg.get("url"),
            auth=(username or cfg.get("username"), password or cfg.get("password")),
            max_connection_pool_size=50,
            connection_timeout=30.0,
        )
        self._verify_connectivity()

    def _verify_connectivity(self) -> None:
        try:
            with self.driver.session(database="neo4j") as session:
                session.run("RETURN 1 AS ok").single()
            logger.info("Neo4j connection verified")
        except Exception as e:
            logger.error(f"Neo4j connectivity failed: {e}")
            raise

    def execute_read(self, query: str, parameters: Dict | None = None) -> List[Dict]:
        with self.driver.session(database="neo4j") as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]

    def execute_write(self, query: str, parameters: Dict | None = None) -> Dict:
        with self.driver.session(database="neo4j") as session:
            return session.execute_write(self._tx_write, query, parameters or {})

    @staticmethod
    def _tx_write(tx: ManagedTransaction, query: str, params: Dict) -> Dict:
        res = tx.run(query, params)
        summary = res.consume()
        return {"counters": summary.counters}

    def create_schema(self) -> None:
        """
        Apply constraints and indexes aligned with our project schema.
        Safe to re-run; IF NOT EXISTS guards handle duplicates.
        """
        ddls = [
            ("constraint", f"CREATE CONSTRAINT knowledge_node_id IF NOT EXISTS FOR (n:KnowledgeNode) REQUIRE n.{Config.PROPERTY_ID} IS UNIQUE"),
            ("constraint", "CREATE CONSTRAINT student_id IF NOT EXISTS FOR (s:Student) REQUIRE s.StudentID IS UNIQUE"),
            ("index", f"CREATE INDEX knowledge_node_context IF NOT EXISTS FOR (n:KnowledgeNode) ON (n.{Config.PROPERTY_CONTEXT})"),
            ("index", f"CREATE INDEX knowledge_node_skill IF NOT EXISTS FOR (n:KnowledgeNode) ON (n.{Config.PROPERTY_SKILL_LEVEL})"),
            ("index", f"CREATE INDEX knowledge_node_priority IF NOT EXISTS FOR (n:KnowledgeNode) ON (n.{Config.PROPERTY_PRIORITY})"),
            ("index", f"CREATE INDEX knowledge_node_difficulty IF NOT EXISTS FOR (n:KnowledgeNode) ON (n.{Config.PROPERTY_DIFFICULTY})"),
            ("index", f"CREATE INDEX knowledge_node_time IF NOT EXISTS FOR (n:KnowledgeNode) ON (n.{Config.PROPERTY_TIME_ESTIMATE})"),
            ("index", f"CREATE INDEX knowledge_node_tags IF NOT EXISTS FOR (n:KnowledgeNode) ON (n.{Config.PROPERTY_SEMANTIC_TAGS})"),
            ("index", "CREATE INDEX student_level IF NOT EXISTS FOR (s:Student) ON (s.current_level)"),
            ("index", "CREATE INDEX student_style IF NOT EXISTS FOR (s:Student) ON (s.learning_style_preference)"),
            ("index", "CREATE INDEX learning_data_student_time IF NOT EXISTS FOR (ld:LearningData) ON (ld.student_id, ld.timestamp)"),
            ("index", "CREATE INDEX learning_data_node IF NOT EXISTS FOR (ld:LearningData) ON (ld.node_id)"),
            ("rel_index", "CREATE INDEX rel_weight IF NOT EXISTS FOR ()-[r]->() ON (r.Weight)"),
            ("rel_index", "CREATE INDEX rel_dependency IF NOT EXISTS FOR ()-[r]->() ON (r.Dependency)"),
        ]

        with self.driver.session(database="neo4j") as session:
            for kind, stmt in ddls:
                try:
                    session.run(stmt)
                    logger.info(f"Applied {kind}: {stmt}")
                except Exception as e:
                    logger.warning(f"Skip {kind}: {e}")

    def close(self) -> None:
        self.driver.close()
