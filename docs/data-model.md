# Data Model

This repository uses a `KnowledgeNode`-centric schema with explicit learner data and interactions.

## Nodes

- KnowledgeNode
  - Required (mapped from Config):
    - Node_ID (`Config.PROPERTY_ID`)
    - Sanitized_Concept (`Config.PROPERTY_SANITIZED_CONCEPT`)
    - Context (`Config.PROPERTY_CONTEXT`)
    - Definition (`Config.PROPERTY_DEFINITION`)
    - Example (`Config.PROPERTY_EXAMPLE`)
    - Learning_Objective (`Config.PROPERTY_LEARNING_OBJECTIVE`)
    - Skill_Level (`Config.PROPERTY_SKILL_LEVEL`) — Bloom levels
    - Time_Estimate (`Config.PROPERTY_TIME_ESTIMATE`) — minutes
    - Difficulty (`Config.PROPERTY_DIFFICULTY`) — STANDARD/ADVANCED
    - Priority (`Config.PROPERTY_PRIORITY`) — 1..5
    - Prerequisites (`Config.PROPERTY_PREREQUISITES`)
    - Semantic_Tags (`Config.PROPERTY_SEMANTIC_TAGS`)
    - Focused_Semantic_Tags (`Config.PROPERTY_FOCUSED_SEMANTIC_TAGS`)

- Student
  - StudentID
  - learning_style_preference
  - current_level
  - time_availability
  - learning_history (CSV)
  - performance_details (semicolon-delimited)

- LearningData
  - student_id
  - node_id
  - timestamp (datetime)
  - score (float)
  - time_spent (int)
  - feedback, quiz_responses

## Relationships

- REQUIRES, IS_PREREQUISITE_OF, NEXT, REMEDIATES, HAS_ALTERNATIVE_PATH, SIMILAR_TO, IS_SUBCONCEPT_OF
  - Relationship properties (optional): Weight (1-5), Dependency (1-5)

- (Student)-[:HAS_LEARNING_DATA]->(LearningData)-[:RELATED_TO_NODE]->(KnowledgeNode)
- (Student)-[:MASTERED]->(KnowledgeNode)
- (Student)-[:MASTERY]->(KnowledgeNode)
  - Properties on MASTERY: score, confidence, lastAssessmentDate, nextOptimalReviewDate, assessmentMethod, previousScore

## Example Cypher

```cypher
MATCH (start:KnowledgeNode {Node_ID: "concept:sql_select"})
MATCH (goal:KnowledgeNode {Node_ID: "concept:group_by"})
MATCH p = shortestPath(
  (start)-[:NEXT|REQUIRES|IS_PREREQUISITE_OF*]->(goal)
)
RETURN p
```

## Indexes & Constraints

- See `backend/scripts/neo4j_schema_setup.py` or `Neo4jManager.create_schema()` for constraints and indexes applied.
