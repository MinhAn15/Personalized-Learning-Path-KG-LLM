from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

from .explainability import llm_complete, llm_model_name

try:
    from .hybrid_retriever import HybridRetriever
except Exception:  # pragma: no cover - optional dependency guard
    HybridRetriever = None  # type: ignore

logger = logging.getLogger(__name__)


def _format_history(history: Sequence[Dict[str, Any]], limit: int = 8) -> str:
    if not history:
        return ""
    trimmed = history[-limit:]
    lines: List[str] = []
    for item in trimmed:
        role = item.get("role", "user").capitalize()
        content = item.get("content", "")
        if not content:
            continue
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _format_supporting_nodes(nodes: Sequence[Dict[str, Any]], limit: int = 5) -> str:
    if not nodes:
        return ""
    captured = nodes[:limit]
    lines: List[str] = []
    for node in captured:
        title = node.get("concept") or node.get("title") or node.get("node_id") or node.get("id")
        if not title:
            continue
        bullet_parts: List[str] = []
        objective = node.get("objective")
        if objective:
            bullet_parts.append(str(objective))
        context = node.get("context")
        if context:
            bullet_parts.append(f"context={context}")
        mastery = node.get("mastery")
        deficiency = node.get("deficiency")
        if mastery is not None:
            bullet_parts.append(f"mastery={mastery:.2f}")
        if deficiency is not None:
            bullet_parts.append(f"gap={deficiency:.2f}")
        if bullet_parts:
            lines.append(f"- {title}: " + ", ".join(bullet_parts))
        else:
            lines.append(f"- {title}")
    return "\n".join(lines)


class ChatService:
    """LLM-backed conversational layer for multi-turn learner coaching."""

    def __init__(self, driver: Optional[Any] = None) -> None:
        self.driver = driver
    self._retriever = None

    def set_driver(self, driver: Optional[Any]) -> None:
        """Attach or replace the Neo4j driver and reset cached retriever."""
        self.driver = driver
        self._retriever = None

    def _ensure_retriever(self):
        if self._retriever is not None:
            return self._retriever
        if HybridRetriever is None or self.driver is None:
            logger.debug("Retriever unavailable (HybridRetriever=%s, driver=%s)", HybridRetriever, self.driver)
            return None
        try:
            self._retriever = HybridRetriever(self.driver)
            return self._retriever
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to initialize HybridRetriever: %s", exc)
            self._retriever = None
            return None

    def _retrieve_context(
        self,
        message: str,
        learner_id: Optional[str],
    ) -> Dict[str, Any]:
        retriever = self._ensure_retriever()
        if retriever is None:
            return {"summary_text": "", "supporting_nodes": [], "context": {}}
        try:
            if learner_id:
                return retriever.retrieve_dynamic_context(message, learner_id)
            return retriever.retrieve_community_summary(message)
        except Exception as exc:  # pragma: no cover - runtime guard
            logger.warning("Retriever failed for message '%s': %s", message, exc)
            return {"summary_text": "", "supporting_nodes": [], "context": {}}

    def _build_prompt(
        self,
        learner_id: Optional[str],
        goal: Optional[str],
        extra_context: Optional[str],
        session_history: Sequence[Dict[str, Any]],
        current_message: str,
        retrieval: Dict[str, Any],
    ) -> str:
        history_block = _format_history(session_history)
        node_block = _format_supporting_nodes(retrieval.get("supporting_nodes", []))
        summary_text = retrieval.get("summary_text") or ""
        context_meta = retrieval.get("context") or {}
        prompt_lines = [
            "You are a personalized learning coach guiding a learner through data/tech topics.",
            "Incorporate knowledge graph insights when helpful and keep responses concise (<= 5 sentences).",
            "Adopt the learner's language from the latest user message.",
            "Encourage strategic study planning and reference specific concepts when appropriate.",
        ]
        if learner_id:
            prompt_lines.append(f"Learner ID: {learner_id}")
        if goal:
            prompt_lines.append(f"Stated goal: {goal}")
        if extra_context:
            prompt_lines.append(f"Learner context: {extra_context}")
        if context_meta:
            prompt_lines.append(f"Retriever context metadata: {context_meta}")
        if summary_text:
            prompt_lines.append(f"Knowledge graph summary: {summary_text}")
        if node_block:
            prompt_lines.append("Relevant nodes:\n" + node_block)
        if history_block:
            prompt_lines.append("Conversation so far:\n" + history_block)
        prompt_lines.append("Current user message: " + current_message)
        prompt_lines.append("Respond with actionable guidance and, when helpful, suggest next topics or resources.")
        return "\n\n".join(prompt_lines)

    def _heuristic_response(
        self,
        message: str,
        retrieval: Dict[str, Any],
        goal: Optional[str],
    ) -> str:
        supporting = retrieval.get("supporting_nodes") or []
        if supporting:
            suggestions = []
            for node in supporting[:3]:
                concept = node.get("concept") or node.get("title") or node.get("node_id")
                if not concept:
                    continue
                detail_bits: List[str] = []
                objective = node.get("objective")
                if objective:
                    detail_bits.append(str(objective))
                mastery = node.get("mastery")
                if mastery is not None:
                    detail_bits.append(f"mastery≈{mastery:.2f}")
                deficiency = node.get("deficiency")
                if deficiency is not None:
                    detail_bits.append(f"focus_gap≈{deficiency:.2f}")
                suggestion = concept
                if detail_bits:
                    suggestion += " (" + ", ".join(detail_bits) + ")"
                suggestions.append(suggestion)
            bullet_list = "\n".join(f"- {s}" for s in suggestions)
            heading = "Dưới đây là các chủ đề nên xem xét:" if any(ord(c) > 127 for c in message) else "Here are focus areas to consider:"
            return f"{heading}\n{bullet_list}\nSử dụng kết hợp lý thuyết và bài tập thực hành để củng cố kiến thức." if heading.startswith("Dưới") else f"{heading}\n{bullet_list}\nBlend conceptual review with hands-on practice to reinforce learning."
        goal_text = goal or "mục tiêu học tập hiện tại"
        return (
            "Tôi sẽ ghi nhớ nhu cầu của bạn và đề xuất lộ trình từng bước khi có dữ liệu cụ thể hơn. "
            "Hãy mô tả chi tiết kỹ năng muốn đạt được hoặc tài nguyên bạn ưu tiên nhé."
            if any(ord(c) > 127 for c in message)
            else f"I'm noting your goal ({goal_text}). Share more about prerequisites or preferred resources so I can tailor a plan."
        )

    def generate_reply(
        self,
        session: Dict[str, Any],
        message: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        metadata = metadata or {}
        payload = session.get("payload", {})
        learner_id = metadata.get("learner_id") or payload.get("learner_id") or payload.get("student_id")
        goal = metadata.get("goal") or payload.get("goal") or payload.get("student_goal")
        extra_context = metadata.get("context") or payload.get("context")

        retrieval = self._retrieve_context(message, learner_id)
        prompt = self._build_prompt(
            learner_id=learner_id,
            goal=goal,
            extra_context=extra_context,
            session_history=session.get("chat_history", []),
            current_message=message,
            retrieval=retrieval,
        )
        llm_text = llm_complete(prompt)
        fallback = False
        if not llm_text:
            fallback = True
            llm_text = self._heuristic_response(message, retrieval, goal)

        sources = retrieval.get("supporting_nodes", [])
        return {
            "reply": llm_text.strip(),
            "supporting_nodes": sources,
            "summary": retrieval.get("summary_text", ""),
            "context": retrieval.get("context", {}),
            "model": llm_model_name(),
            "fallback": fallback,
        }
