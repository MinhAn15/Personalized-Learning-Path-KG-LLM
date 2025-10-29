# Architecture Overview (C4-lite)

This document clarifies the current architecture and how components map to the repository.

## Context

- Goal: Personalized learning path generation using a Knowledge Graph (Neo4j) + LLM assistance.
- Actors: Learner, Educator, Backend Service, Neo4j, Optional Frontend (Next.js or Streamlit).

## Containers

- Backend (Python)
  - FastAPI app (see `backend/src/api.py`)
  - Core logic modules under `backend/src/` (path generation, learner state, tracing, retrieval)
- Database: Neo4j (AuraDB/Desktop)
  - Label `KnowledgeNode` and relationships (REQUIRES, NEXT, IS_PREREQUISITE_OF, ...)
  - `Student` and `LearningData` for learner events and mastery
- Frontend: Next.js (optional)
  - Demo UI in `frontend/`

## Components (current mapping)

- Graph access & schema
  - `backend/scripts/neo4j_schema_setup.py` (CLI)
  - `backend/src/neo4j_manager.py` (pooled manager + create_schema())
- Data loading & utilities
  - `backend/src/data_loader.py` (CSV import, utility queries)
- Recommendation logic
  - `backend/src/path_generator.py` (start/goal, heuristic, A*)
  - `backend/src/adaptive_path_planner.py` (wrapper adapter; optional)
  - `backend/src/hybrid_retriever.py` (Graph RAG + tag-sim surrogate)
- Learner modeling and tracing
  - `backend/src/learner_state.py` (mastery, review schedule, from_neo4j)
  - `backend/src/knowledge_tracing.py` (decay + Bayesian-like update to MASTERY)
  - `backend/src/learner_profile_manager.py` (Student CRUD and summary)
- Explainability & quality tools
  - `backend/src/explainability.py` (path explanations & counterfactuals)
  - `backend/src/kg_quality.py` (light KG checks)
- LLM & prompts
  - `prompts/` (SPR generator/validation templates)

## Data Flow (Phase 1/2)

1) Phase 1 — Build KG
- Prepare `master_nodes.csv` + `master_relationships.csv`
- Run `backend/scripts/neo4j_schema_setup.py`
- Import via `backend/src/data_loader.check_and_load_kg()` or `scripts/import_data.py`

2) Phase 2 — Recommend Paths
- Backend receives learner goal/context
- Determine start/goal (path_generator)
- Run A* with current heuristic and dynamic weights
- Return path; add explanations; update learner state after assessments

## Notes

- Vector retrieval can be plugged in later (Neo4j vector index); current surrogate uses tag similarity.
- LLM usage is kept optional and routed via existing wrappers.
