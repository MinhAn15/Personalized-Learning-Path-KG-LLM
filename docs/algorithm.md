# Path Planning Algorithm

This document explains the A* search and heuristic currently used, with room for future optimization.

## Inputs
- Start node (KnowledgeNode.Node_ID)
- Goal node (KnowledgeNode.Node_ID)
- Learner profile (Student + LearningData)
- Dynamic weights (derived from learner state and config)

## Heuristic Function (current)
The heuristic blends multiple factors:
- Semantic similarity to goal (Jaccard over tags)
- Priority (higher priority increases cost modestly)
- Difficulty (STANDARD vs ADVANCED)
- Skill level (Bloom level scalar)
- Time estimate (minutes), scaled by config and dynamic time weight
- Context match bonus (reduces cost if context aligns)

Example (as implemented in `backend/src/path_generator.py`):
```
h = 0.4 * (1 - similarity)
  + 0.1 * priority / 5
  + 0.2 * difficulty_weight / 3
  + 0.2 * skill_weight / 6
  + time_estimate * Config.ASTAR_HEURISTIC_WEIGHTS["time_estimate"] * weight_time_estimate
  - context_match
```

Where:
- `similarity ∈ [0,1]` via Jaccard on Semantic_Tags
- `difficulty_weight` = weight for STANDARD or ADVANCED from dynamic_weights
- `skill_weight` = scalar per Bloom level
- `weight_time_estimate` = learner-specific time sensitivity (e.g., slower learners value time more)

## A* Outline
- Evaluate f(n) = g(n) + h(n) with a priority queue
- g(n) accumulates relationship weights (or 1.0 if absent)
- h(n) is the heuristic above
- Stop when goal reached or frontier exhausted

## Edge Cases & Guards
- Missing node properties are coalesced to defaults (priority 0, time 30, etc.)
- Heuristic results are cached per node for speed
- Context match adds a small negative term (bonus) for aligned contexts

## Future Work
- Replace tag-similarity with vector embeddings (Neo4j vector index)
- Multi-objective tuning via Optuna (time vs similarity vs success probability)
- Add path length penalties or prerequisite depth terms when needed
- Integrate knowledge tracing outputs for goal feasibility
