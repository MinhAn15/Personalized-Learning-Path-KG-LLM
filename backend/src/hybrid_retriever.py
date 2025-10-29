from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import re

from .config import Config
from .data_loader import execute_cypher_query, jaccard_similarity


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    text = text.lower()
    # split on non-alphanumeric and underscore
    parts = re.split(r"[^a-z0-9_]+", text)
    return [p for p in parts if p]


class HybridRetriever:
    """
    Hybrid retriever that routes between Graph RAG (structure-aware) and a
    lightweight similarity retriever (tag-based surrogate for vectors).

    This keeps dependencies minimal while providing a clean seam to integrate
    a true vector index/embedding model later (e.g., Neo4j vector index or LlamaIndex).
    """

    def __init__(self, neo4j_driver: Any) -> None:
        self.driver = neo4j_driver

    # --- Graph RAG ------------------------------------------------------------
    def graph_retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve structurally relevant nodes based on prerequisite connectivity.
        A simple pattern: find high-priority leaves and near-prereqs matching tokens.
        """
        tokens = _tokenize(query)
        if not tokens:
            tokens = [""]

        # Parameterized query; match tags or objective containing tokens
        cypher = f"""
        WITH $tokens AS toks
        MATCH (n:KnowledgeNode)
        WHERE any(t IN toks WHERE toLower(n.{Config.PROPERTY_LEARNING_OBJECTIVE}) CONTAINS t)
            OR any(t IN toks WHERE toLower(n.{Config.PROPERTY_SANITIZED_CONCEPT}) CONTAINS t)
            OR any(t IN toks WHERE t IN split(toLower(coalesce(n.{Config.PROPERTY_SEMANTIC_TAGS}, '')), ';'))
        OPTIONAL MATCH (n)-[r:{Config.RELATIONSHIP_IS_PREREQUISITE_OF}|{Config.RELATIONSHIP_NEXT}]->(:KnowledgeNode)
        WITH n, coalesce(n.{Config.PROPERTY_PRIORITY},0) AS pri, count(r) AS outs
        RETURN n.{Config.PROPERTY_ID} AS id,
               n.{Config.PROPERTY_SANITIZED_CONCEPT} AS concept,
               n.{Config.PROPERTY_LEARNING_OBJECTIVE} AS objective,
               n.{Config.PROPERTY_CONTEXT} AS context,
               n.{Config.PROPERTY_PRIORITY} AS priority,
               n.{Config.PROPERTY_TIME_ESTIMATE} AS time_estimate,
               n.{Config.PROPERTY_DIFFICULTY} AS difficulty,
               outs AS out_degree
        ORDER BY pri DESC, outs ASC
        LIMIT $k
        """
        rows = execute_cypher_query(self.driver, cypher, {"tokens": tokens, "k": int(top_k)})
        return rows

    # --- Vector-like RAG (surrogate) -----------------------------------------
    def tag_similarity_retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Approximate semantic retrieval via Jaccard similarity over semantic tags.
        This mimics vector search behavior without requiring embeddings.
        """
        tokens = _tokenize(query)
        token_set = set(tokens)
        cypher = f"""
        MATCH (n:KnowledgeNode)
        RETURN n.{Config.PROPERTY_ID} AS id,
               n.{Config.PROPERTY_SANITIZED_CONCEPT} AS concept,
               n.{Config.PROPERTY_LEARNING_OBJECTIVE} AS objective,
               n.{Config.PROPERTY_CONTEXT} AS context,
               n.{Config.PROPERTY_PRIORITY} AS priority,
               n.{Config.PROPERTY_TIME_ESTIMATE} AS time_estimate,
               n.{Config.PROPERTY_DIFFICULTY} AS difficulty,
               coalesce(n.{Config.PROPERTY_SEMANTIC_TAGS}, '') AS tags
        """
        rows = execute_cypher_query(self.driver, cypher)
        rescored: List[Tuple[float, Dict[str, Any]]] = []
        for r in rows:
            tags = [t.strip().lower() for t in (r.get("tags") or "").split(";") if t.strip()]
            sim = jaccard_similarity(list(token_set), tags)
            rescored.append((sim, r))
        rescored.sort(key=lambda x: (x[0], float(x[1].get("priority") or 0.0)), reverse=True)
        return [r for _, r in rescored[: max(0, int(top_k))]]

    # --- Router ---------------------------------------------------------------
    def retrieve(self, query: str, learner_id: Optional[str] = None, context_type: Optional[str] = None, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Route to Graph or tag-similarity retriever based on context_type.
        If unspecified, use a trivial heuristic: structural keywords -> graph; else tags.
        """
        if context_type == "prerequisite_structure":
            return self.graph_retrieve(query, top_k)
        if context_type == "similar_materials":
            return self.tag_similarity_retrieve(query, top_k)

        # Simple heuristic router
        structure_terms = {"prereq", "prerequisite", "path", "sequence", "order", "dependency"}
        tokens = set(_tokenize(query))
        if tokens & structure_terms:
            return self.graph_retrieve(query, top_k)
        return self.tag_similarity_retrieve(query, top_k)
