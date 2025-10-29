from __future__ import annotations

from typing import Any, Dict, Optional
import logging

from .neo4j_manager import Neo4jManager
from .config import Config

logger = logging.getLogger(__name__)


class LearnerProfileManager:
    """
    Manage Student profiles in Neo4j (create/update/retrieve summaries).
    Tailored to current schema: :Student nodes and :LearningData events.
    """

    def __init__(self, neo4j: Neo4jManager) -> None:
        self.neo4j = neo4j

    def create_student(self, student_id: str, initial_profile: Optional[Dict] = None) -> Dict:
        """Create a Student node with basic attributes if it doesn't exist."""
        initial_profile = initial_profile or {}
        q = """
        MERGE (s:Student {StudentID: $sid})
        ON CREATE SET s.createdAt = datetime(),
                      s.learning_style_preference = coalesce($style, $default_style),
                      s.current_level = toFloat(coalesce($level, 0)),
                      s.time_availability = toFloat(coalesce($hours, 60))
        RETURN s
        """
        params = {
            "sid": student_id,
            "style": initial_profile.get("learning_style"),
            "default_style": Config.DEFAULT_LEARNING_STYLE,
            "level": initial_profile.get("knowledge_level", 0),
            "hours": initial_profile.get("available_hours", 60),
        }
        return self.neo4j.execute_write(q, params)

    def update_profile_dimension(self, student_id: str, dimension: str, updates: Dict) -> Dict:
        """Upsert a map property for a given dimension on Student."""
        if not isinstance(updates, dict):
            raise ValueError("updates must be a dictionary")
        q = f"""
        MERGE (s:Student {{StudentID: $sid}})
        SET s.{dimension} = $updates,
            s.lastUpdated = datetime()
        RETURN s
        """
        return self.neo4j.execute_write(q, {"sid": student_id, "updates": updates})

    def get_student_profile(self, student_id: str) -> Optional[Dict]:
        """Return Student summary with a few derived metrics."""
        q = f"""
        MATCH (s:Student {{StudentID: $sid}})
        OPTIONAL MATCH (s)-[:MASTERED]->(kn:KnowledgeNode)
        WITH s, count(kn) AS mastered_count
        OPTIONAL MATCH (s)-[:HAS_LEARNING_DATA]->(ld:LearningData)
        RETURN s{{.*, mastered_count: mastered_count}} AS profile,
               avg(ld.score) AS avg_score,
               count(ld) AS interactions
        """
        rows = self.neo4j.execute_read(q, {"sid": student_id})
        if not rows:
            return None
        row = rows[0]
        profile = row.get("profile") or {}
        profile["avg_ld_score"] = float(row.get("avg_score") or 0.0)
        profile["interactions_30d"] = int(row.get("interactions") or 0)
        return profile
