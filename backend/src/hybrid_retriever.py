from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .config import Config
from .data_loader import execute_cypher_query, jaccard_similarity
from .learner_state import LearnerState

logger = logging.getLogger(__name__)

try:  # Optional scientific stack for community detection
    import networkx as nx  # type: ignore
except ImportError:  # pragma: no cover - handled at runtime
    nx = None  # type: ignore

try:  # Louvain clustering (python-louvain)
    from community import community_louvain  # type: ignore
except ImportError:  # pragma: no cover - handled at runtime
    community_louvain = None  # type: ignore

try:  # LlamaIndex components (GraphRAG and vector retrieval)
    from llama_index.core import StorageContext, load_index_from_storage  # type: ignore
    from llama_index.core.indices.property_graph import PropertyGraphIndex  # type: ignore
    from llama_index.core.retrievers import VectorIndexRetriever  # type: ignore
    from llama_index.core.schema import NodeWithScore  # type: ignore
    from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore  # type: ignore
except ImportError:  # pragma: no cover - degrade gracefully
    StorageContext = None  # type: ignore
    load_index_from_storage = None  # type: ignore
    PropertyGraphIndex = None  # type: ignore
    VectorIndexRetriever = None  # type: ignore
    NodeWithScore = Any  # type: ignore
    Neo4jPropertyGraphStore = None  # type: ignore

try:
    from llama_index.retrievers import KnowledgeGraphRetriever  # type: ignore
except ImportError:  # pragma: no cover - degrade gracefully
    KnowledgeGraphRetriever = None  # type: ignore


GRAPH_TOP_K = 12
VECTOR_TOP_K = 8
DEFAULT_COMMUNITY_ID = 0


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    text = text.lower()
    parts = re.split(r"[^a-z0-9_]+", text)
    return [p for p in parts if p]


@dataclass
class RetrievedNode:
    node_id: str
    concept: str = ""
    objective: str = ""
    context: str = ""
    text: str = ""
    source: str = "unknown"
    score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    community_id: Optional[int] = None
    mastery: Optional[float] = None
    deficiency: Optional[float] = None
    priority_rank: Optional[int] = None
    sources: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.source and self.source not in self.sources:
            self.sources.append(self.source)

    def update_with(self, other: "RetrievedNode") -> None:
        self.score = self._prefer_score(self.score, other.score)
        self.metadata.update(other.metadata)
        self.text = self.text or other.text
        self.concept = self.concept or other.concept
        self.objective = self.objective or other.objective
        self.context = self.context or other.context
        for src in other.sources:
            if src not in self.sources:
                self.sources.append(src)

    @staticmethod
    def _prefer_score(current: Optional[float], candidate: Optional[float]) -> Optional[float]:
        if candidate is None:
            return current
        if current is None:
            return candidate
        return candidate if candidate > current else current

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "node_id": self.node_id,
            "concept": self.concept,
            "objective": self.objective,
            "context": self.context,
            "source": self.source,
            "score": self.score,
            "community_id": self.community_id,
            "mastery": self.mastery,
            "deficiency": self.deficiency,
            "priority_rank": self.priority_rank,
            "sources": sorted(self.sources),
        }
        if self.text:
            payload["text"] = self.text
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload


class HybridRetriever:
    """GraphRAG retriever combining Neo4j structural context with vector recall."""

    def __init__(
        self,
        neo4j_driver: Any,
        graph_top_k: int = GRAPH_TOP_K,
        vector_top_k: int = VECTOR_TOP_K,
    ) -> None:
        self.driver = neo4j_driver
        self.graph_top_k = graph_top_k
        self.vector_top_k = vector_top_k

        self._kg_retriever: Optional[Any] = None
        self._vector_retriever: Optional[VectorIndexRetriever] = None
        self._graph_store: Optional[Any] = None
        self._kg_index: Optional[Any] = None

        self._initialise_graph_retriever()
        self._initialise_vector_retriever()

    def _initialise_graph_retriever(self) -> None:
        if KnowledgeGraphRetriever is None or PropertyGraphIndex is None or Neo4jPropertyGraphStore is None:
            logger.warning("KnowledgeGraphRetriever unavailable; GraphRAG fallback to legacy graph queries.")
            return

        try:
            graph_store = Neo4jPropertyGraphStore(
                username=Config.NEO4J_CONFIG.get("username"),
                password=Config.NEO4J_CONFIG.get("password"),
                url=Config.NEO4J_CONFIG.get("url"),
                database="neo4j",
            )
            if os.path.isdir(Config.PROPERTY_GRAPH_STORAGE_DIR):
                kg_index = PropertyGraphIndex.from_existing(
                    property_graph_store=graph_store,
                    persist_dir=Config.PROPERTY_GRAPH_STORAGE_DIR,
                )
            else:
                kg_index = PropertyGraphIndex.from_existing(
                    property_graph_store=graph_store,
                    embed_kg_nodes=True,
                    include_embeddings=True,
                )
                kg_index.storage_context.persist(persist_dir=Config.PROPERTY_GRAPH_STORAGE_DIR)

            retriever = kg_index.as_retriever(include_text=True)
            if hasattr(retriever, "similarity_top_k"):
                retriever.similarity_top_k = self.graph_top_k

            self._kg_retriever = retriever
            self._graph_store = graph_store
            self._kg_index = kg_index
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Failed to initialise property-graph retriever: %s", exc)
            self._kg_retriever = None

    def _initialise_vector_retriever(self) -> None:
        if StorageContext is None or load_index_from_storage is None:
            logger.warning("Vector retrieval unavailable (StorageContext missing).")
            return
        if not os.path.isdir(Config.VECTOR_INDEX_STORAGE_DIR):
            logger.info("Vector index storage not found at %s", Config.VECTOR_INDEX_STORAGE_DIR)
            return

        try:
            storage_ctx = StorageContext.from_defaults(persist_dir=Config.VECTOR_INDEX_STORAGE_DIR)
            try:
                vector_index = load_index_from_storage(storage_ctx)
            except Exception:
                vector_index = load_index_from_storage(storage_ctx, index_id="vector_index")
            retriever = vector_index.as_retriever(similarity_top_k=self.vector_top_k)
            self._vector_retriever = retriever
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to initialise vector retriever: %s", exc)
            self._vector_retriever = None

    def retrieve(
        self,
        query: str,
        learner_id: Optional[str] = None,
        context_type: Optional[str] = None,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        if learner_id and context_type not in {"community_summary", "prerequisite_structure"}:
            result = self.retrieve_dynamic_context(query, learner_id)
        else:
            result = self.retrieve_community_summary(query)

        nodes = result["supporting_nodes"]
        return nodes[:top_k] if top_k else nodes

    def retrieve_community_summary(self, query: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        prepared = self._prepare_graph_context(query, top_k or self.graph_top_k)
        nodes, graph, summary_text, context = prepared
        supporting_nodes = [node.to_dict() for node in self._sorted_nodes(nodes)]
        return {
            "context": context,
            "supporting_nodes": supporting_nodes,
            "summary_text": summary_text,
        }

    def retrieve_dynamic_context(
        self,
        query: str,
        learner_id: str,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        nodes, graph, base_summary, context = self._prepare_graph_context(query, top_k or self.graph_top_k)
        if not nodes:
            return {
                "context": {"learner_id": learner_id, **context},
                "supporting_nodes": [],
                "summary_text": "No matching context available.",
            }

        learner_state = LearnerState.from_neo4j(self.driver, learner_id)
        for node in nodes:
            mastery = learner_state.get_mastery(node.node_id, 0.0)
            node.mastery = mastery
            node.deficiency = 1.0 - mastery

        prioritized = sorted(
            nodes,
            key=lambda n: (
                n.deficiency if n.deficiency is not None else -1.0,
                n.score or 0.0,
            ),
            reverse=True,
        )
        for rank, node in enumerate(prioritized, start=1):
            node.priority_rank = rank

        focus_nodes = prioritized[:3]
        focus_summary = ", ".join(
            f"{node.concept or node.node_id} ({node.mastery:.2f})" for node in focus_nodes
            if node.mastery is not None
        )
        learner_context = {
            "learner_id": learner_id,
            "time_budget": learner_state.time_budget_minutes,
            "preferences": learner_state.preferences,
            "focus_nodes": [
                {
                    "node_id": node.node_id,
                    "deficiency": node.deficiency,
                    "mastery": node.mastery,
                    "priority_rank": node.priority_rank,
                }
                for node in focus_nodes
            ],
        }
        context.update(learner_context)

        dynamic_summary = (
            f"Learner {learner_id} priorities: {focus_summary}" if focus_summary else "No standout priorities for learner."
        )
        summary_text = f"{base_summary} || {dynamic_summary}"

        supporting_nodes = [node.to_dict() for node in prioritized]
        if top_k:
            supporting_nodes = supporting_nodes[:top_k]

        return {
            "context": context,
            "supporting_nodes": supporting_nodes,
            "summary_text": summary_text,
        }

    def _prepare_graph_context(
        self,
        query: str,
        graph_top_k: int,
    ) -> Tuple[List[RetrievedNode], Optional[Any], str, Dict[str, Any]]:
        graph_nodes = self._graph_search_nodes(query, graph_top_k)
        vector_nodes = self._vector_search_nodes(query, self.vector_top_k)
        combined = self._merge_nodes(graph_nodes, vector_nodes)

        if not combined:
            return combined, None, "No communities matched the query.", {
                "communities": [],
                "node_count": 0,
                "edge_count": 0,
            }

        graph = self._build_graph(combined)
        partition = self._detect_communities(graph) if graph is not None else {
            node.node_id: DEFAULT_COMMUNITY_ID for node in combined
        }
        for node in combined:
            node.community_id = partition.get(node.node_id, DEFAULT_COMMUNITY_ID)

        summary_text, context = self._summarise_communities(combined, graph)
        return combined, graph, summary_text, context

    def _graph_search_nodes(self, query: str, top_k: int) -> List[RetrievedNode]:
        if self._kg_retriever is None:
            return self._legacy_graph_search(query, top_k)
        if hasattr(self._kg_retriever, "similarity_top_k"):
            self._kg_retriever.similarity_top_k = top_k
        try:
            results = self._kg_retriever.retrieve(query)
            return self._normalise_results(results, source="graph_rag")
        except Exception as exc:  # pragma: no cover
            logger.warning("KnowledgeGraphRetriever failed (%s); using legacy fallback.", exc)
            return self._legacy_graph_search(query, top_k)

    def _vector_search_nodes(self, query: str, top_k: int) -> List[RetrievedNode]:
        if self._vector_retriever is None:
            return self._legacy_tag_search(query, top_k)
        if hasattr(self._vector_retriever, "similarity_top_k"):
            self._vector_retriever.similarity_top_k = top_k
        try:
            results = self._vector_retriever.retrieve(query)
            return self._normalise_results(results, source="vector")
        except Exception as exc:  # pragma: no cover
            logger.warning("Vector retriever failed (%s); using legacy tag similarity.", exc)
            return self._legacy_tag_search(query, top_k)

    def _normalise_results(self, results: Iterable[NodeWithScore], source: str) -> List[RetrievedNode]:
        normalised: List[RetrievedNode] = []
        for item in results or []:
            node = self._node_from_result(item, source)
            if node:
                normalised.append(node)
        return normalised

    def _node_from_result(self, result: Any, source: str) -> Optional[RetrievedNode]:
        if result is None:
            return None

        node_obj = getattr(result, "node", None)
        metadata = dict(getattr(node_obj, "metadata", {}) or getattr(result, "metadata", {}) or {})
        text = getattr(node_obj, "text", None) or getattr(result, "text", "")
        score = getattr(result, "score", None)

        node_id = (
            metadata.get(Config.PROPERTY_ID)
            or metadata.get("node_id")
            or metadata.get("NodeID")
            or metadata.get("id")
            or getattr(node_obj, "node_id", None)
            or getattr(result, "id", None)
        )
        concept = metadata.get(Config.PROPERTY_SANITIZED_CONCEPT) or metadata.get("concept")
        objective = metadata.get(Config.PROPERTY_LEARNING_OBJECTIVE) or metadata.get("objective")
        context_value = metadata.get(Config.PROPERTY_CONTEXT) or metadata.get("context")

        if not node_id:
            if concept:
                node_id = f"{source}:{concept}"
            elif text:
                node_id = f"{source}:{abs(hash(text))}"
            else:
                return None

        return RetrievedNode(
            node_id=str(node_id),
            concept=concept or "",
            objective=objective or "",
            context=context_value or "",
            text=text or "",
            score=float(score) if score is not None else None,
            metadata=metadata,
            source=source,
        )

    def _merge_nodes(self, *node_lists: Iterable[RetrievedNode]) -> List[RetrievedNode]:
        merged: Dict[str, RetrievedNode] = {}
        for nodes in node_lists:
            for node in nodes:
                existing = merged.get(node.node_id)
                if existing:
                    existing.update_with(node)
                else:
                    merged[node.node_id] = node
        return list(merged.values())

    def _build_graph(self, nodes: List[RetrievedNode]) -> Optional[Any]:
        if nx is None:
            logger.debug("networkx not available; skipping subgraph construction.")
            return None
        graph = nx.Graph()
        for node in nodes:
            graph.add_node(node.node_id, concept=node.concept, objective=node.objective)

        node_ids = [node.node_id for node in nodes if node.node_id]
        edges = self._fetch_subgraph_edges(node_ids)
        for edge in edges:
            graph.add_edge(edge["source"], edge["target"], relation=edge.get("relation"))
        return graph

    def _fetch_subgraph_edges(self, node_ids: List[str]) -> List[Dict[str, Any]]:
        if not node_ids:
            return []
        query = f"""
        MATCH (a:KnowledgeNode)-[r]->(b:KnowledgeNode)
        WHERE a.{Config.PROPERTY_ID} IN $ids AND b.{Config.PROPERTY_ID} IN $ids
        RETURN a.{Config.PROPERTY_ID} AS source,
               b.{Config.PROPERTY_ID} AS target,
               type(r) AS relation
        """
        try:
            return execute_cypher_query(self.driver, query, {"ids": node_ids})
        except Exception as exc:  # pragma: no cover
            logger.debug("Subgraph edge retrieval failed: %s", exc)
            return []

    def _detect_communities(self, graph: Any) -> Dict[str, int]:
        if graph is None or getattr(graph, "number_of_nodes", lambda: 0)() == 0:
            return {}
        if community_louvain is None or getattr(graph, "number_of_edges", lambda: 0)() == 0:
            return {node: DEFAULT_COMMUNITY_ID for node in graph.nodes()}
        try:
            partition = community_louvain.best_partition(graph)
            return {str(node): int(cid) for node, cid in partition.items()}
        except Exception as exc:  # pragma: no cover
            logger.debug("Community detection failed: %s", exc)
            return {node: DEFAULT_COMMUNITY_ID for node in graph.nodes()}

    def _summarise_communities(
        self,
        nodes: List[RetrievedNode],
        graph: Optional[Any],
    ) -> Tuple[str, Dict[str, Any]]:
        communities: Dict[int, List[RetrievedNode]] = {}
        for node in nodes:
            cid = node.community_id if node.community_id is not None else DEFAULT_COMMUNITY_ID
            communities.setdefault(cid, []).append(node)

        lines: List[str] = []
        community_payload: List[Dict[str, Any]] = []
        for cid, members in sorted(communities.items(), key=lambda item: item[0]):
            highlight = ", ".join(filter(None, [member.concept or member.node_id for member in members][:5]))
            lines.append(f"Group {cid + 1}: {highlight if highlight else 'undetermined'}")
            community_payload.append(
                {
                    "community_id": int(cid),
                    "size": len(members),
                    "members": [member.node_id for member in members],
                }
            )

    summary_text = " | ".join(lines) if lines else "No meaningful community detected."
        context = {
            "communities": community_payload,
            "node_count": len(nodes),
            "edge_count": getattr(graph, "number_of_edges", lambda: 0)(),
        }
        return summary_text, context

    def _sorted_nodes(self, nodes: List[RetrievedNode]) -> List[RetrievedNode]:
        return sorted(
            nodes,
            key=lambda n: (
                n.community_id if n.community_id is not None else DEFAULT_COMMUNITY_ID,
                -(n.score or 0.0),
                n.node_id,
            ),
        )

    def _legacy_graph_search(self, query: str, top_k: int) -> List[RetrievedNode]:
        tokens = _tokenize(query) or [""]
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
        nodes: List[RetrievedNode] = []
        for row in rows:
            node = RetrievedNode(
                node_id=str(row.get("id")),
                concept=row.get("concept", ""),
                objective=row.get("objective", ""),
                context=row.get("context", ""),
                metadata=row,
                source="graph_legacy",
            )
            nodes.append(node)
        return nodes

    def _legacy_tag_search(self, query: str, top_k: int) -> List[RetrievedNode]:
        tokens = set(_tokenize(query))
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
        rescored: List[Tuple[float, RetrievedNode]] = []
        for row in rows:
            tags = [t.strip().lower() for t in (row.get("tags") or "").split(";") if t.strip()]
            sim = jaccard_similarity(list(tokens), tags)
            node = RetrievedNode(
                node_id=str(row.get("id")),
                concept=row.get("concept", ""),
                objective=row.get("objective", ""),
                context=row.get("context", ""),
                metadata=row,
                score=sim,
                source="vector_legacy",
            )
            rescored.append((sim, node))
        rescored.sort(key=lambda item: (item[0], float(item[1].metadata.get("priority") or 0.0)), reverse=True)
        return [node for _, node in rescored[: max(0, int(top_k))]]
