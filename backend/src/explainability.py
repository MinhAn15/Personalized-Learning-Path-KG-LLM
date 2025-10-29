from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .config import Config

logger = logging.getLogger(__name__)

_LLM_CLIENT: Optional[Any] = None
_LLM_MODEL_NAME: str = "heuristic"


def explain_node_choice(node: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> str:
    """Return a concise heuristic argument for selecting ``node``."""
    title = node.get("title") or node.get("name") or node.get("concept") or node.get("id", "node")
    sim = node.get("similarity")
    pri = node.get("priority")
    diff = node.get("difficulty")
    tmin = node.get("time_estimate")

    pieces: List[str] = []
    if sim:
        try:
            pieces.append(f"high content match ({float(sim):.2f})")
        except Exception:
            pieces.append(f"content match {sim}")
    if pri not in (None, ""):
        pieces.append(f"priority {pri}")
    if diff not in (None, ""):
        pieces.append(f"difficulty {diff}")
    if tmin not in (None, ""):
        try:
            pieces.append(f"~{int(round(float(tmin)))} min")
        except Exception:
            pieces.append(f"time {tmin}")

    if not pieces:
        return f"Selected {title} based on graph structure and prerequisite coverage."
    return f"Selected {title} due to " + ", ".join(pieces) + "."


def explain_path(
    nodes: Sequence[Dict[str, Any]],
    metrics: Optional[Dict[str, Any]] = None,
    learner: Optional[Any] = None,
) -> Dict[str, Any]:
    """Provide a structured heuristic explanation for a learning path."""
    node_list = list(nodes or [])
    steps = [explain_node_choice(n, metrics) for n in node_list]

    total_time = sum(int(float(n.get("time_estimate", 0) or 0)) for n in node_list)
    avg_diff = 0.0
    if node_list:
        try:
            avg_diff = sum(float(n.get("difficulty", 0) or 0.0) for n in node_list) / float(len(node_list))
        except Exception:
            avg_diff = 0.0

    summary_bits = [
        f"Path length: {len(node_list)} nodes",
        f"Estimated time: ~{total_time} minutes",
        f"Average difficulty: {avg_diff:.2f}",
    ]
    if learner and getattr(learner, "time_budget_minutes", None):
        budget = learner.time_budget_minutes
        if budget is not None and total_time > budget:
            summary_bits.append(
                f"Note: path exceeds learner time budget ({total_time}>{budget} min)."
            )

    return {
        "summary": "; ".join(summary_bits),
        "steps": steps,
        "metrics": metrics or {},
        "caveats": [
            "Heuristic scores approximate learner fit; validate with feedback.",
            "Prerequisite coverage should be confirmed before delivery.",
        ],
    }


def _get_llm_client() -> Optional[Any]:
    global _LLM_CLIENT, _LLM_MODEL_NAME
    if _LLM_CLIENT is not None:
        return _LLM_CLIENT

    llm = getattr(Config, "LLM", None)
    if llm is not None:
        _LLM_CLIENT = llm
        _LLM_MODEL_NAME = getattr(llm, "model_name", "config-llm")
        return _LLM_CLIENT

    if getattr(Config, "GEMINI_API_KEY", None):
        try:
            from .genai_wrapper import GenAIWrapper  # type: ignore

            model_name = os.getenv("GEMINI_MODEL", "models/gemini-1.5-flash")
            wrapper = GenAIWrapper(model_name, Config.GEMINI_API_KEY)
            _LLM_CLIENT = wrapper
            _LLM_MODEL_NAME = model_name
            return _LLM_CLIENT
        except Exception as exc:  # pragma: no cover - optional dependency
            logger.debug("Gemini wrapper unavailable: %s", exc)

    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        preferred_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        try:
            from openai import OpenAI  # type: ignore

            class _OpenAIWrapper:
                def __init__(self, key: str, model: str):
                    self._client = OpenAI(api_key=key)
                    self.model = model

                def complete(self, prompt: str):
                    response = self._client.responses.create(model=self.model, input=prompt)
                    text = getattr(response, "output_text", None)
                    if text is None:
                        try:
                            text = "".join(
                                getattr(item, "text", str(item)) for item in getattr(response, "output", [])
                            )
                        except Exception:
                            text = str(response)
                    return type("Resp", (), {"text": text})

            wrapper = _OpenAIWrapper(api_key, preferred_model)
            _LLM_CLIENT = wrapper
            _LLM_MODEL_NAME = preferred_model
            return _LLM_CLIENT
        except Exception as exc:
            logger.debug("OpenAI (responses API) unavailable: %s", exc)
            try:
                import openai  # type: ignore

                openai.api_key = api_key

                class _OpenAILegacyWrapper:
                    def __init__(self, model: str):
                        self.model = model

                    def complete(self, prompt: str):
                        chat = openai.ChatCompletion.create(
                            model=self.model,
                            messages=[
                                {"role": "system", "content": "You are an educational planning assistant."},
                                {"role": "user", "content": prompt},
                            ],
                            temperature=0.3,
                        )
                        text = chat["choices"][0]["message"]["content"]
                        return type("Resp", (), {"text": text})

                wrapper = _OpenAILegacyWrapper(os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"))
                _LLM_CLIENT = wrapper
                _LLM_MODEL_NAME = wrapper.model
                return _LLM_CLIENT
            except Exception as exc2:  # pragma: no cover - optional dependency
                logger.debug("OpenAI (legacy) unavailable: %s", exc2)

    _LLM_MODEL_NAME = "heuristic"
    return None


def _call_llm(prompt: str) -> Optional[str]:
    client = _get_llm_client()
    if client is None:
        return None
    try:
        if hasattr(client, "complete"):
            response = client.complete(prompt)
        elif hasattr(client, "generate_content"):
            response = client.generate_content(prompt)  # type: ignore[attr-defined]
        else:
            logger.debug("Unknown LLM client interface: %s", type(client))
            return None
    except Exception as exc:  # pragma: no cover - runtime guard
        logger.warning("LLM completion failed: %s", exc)
        return None

    if isinstance(response, str):
        return response.strip()
    text = getattr(response, "text", None)
    if text:
        return str(text).strip()
    return str(response).strip() if response is not None else None


def _normalize_path(path: Any) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    metrics: Dict[str, Any] = {}
    if isinstance(path, dict):
        nodes = path.get("nodes") or path.get("path") or path.get("steps") or []
        if isinstance(nodes, dict):
            nodes = list(nodes.values())
        metrics = path.get("metrics") or path.get("meta") or {}
    else:
        nodes = path
    node_list: List[Dict[str, Any]] = []
    for item in list(nodes or []):
        if isinstance(item, dict):
            node_list.append(item)
        else:
            node_list.append({"title": str(item)})
    return node_list, metrics


def _summarize_nodes(nodes: Sequence[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for idx, node in enumerate(nodes, start=1):
        title = node.get("title") or node.get("name") or node.get("concept") or node.get("id") or f"Node {idx}"
        extras: List[str] = []
        priority = node.get("priority")
        if priority not in (None, ""):
            extras.append(f"priority {priority}")
        diff = node.get("difficulty")
        if diff not in (None, ""):
            extras.append(f"difficulty {diff}")
        mastery = node.get("mastery")
        if mastery not in (None, ""):
            try:
                extras.append(f"mastery {float(mastery):.2f}")
            except Exception:
                extras.append(f"mastery {mastery}")
        time_est = node.get("time_estimate")
        if time_est not in (None, ""):
            try:
                extras.append(f"~{int(round(float(time_est)))} min")
            except Exception:
                extras.append(f"time {time_est}")
        context = node.get("context")
        if context:
            extras.append(f"context {context}")
        line = f"{idx}. {title}"
        if extras:
            line += " (" + ", ".join(extras) + ")"
        lines.append(line)
    return "\n".join(lines)


def _heuristic_alternatives(nodes: List[Dict[str, Any]], limit: int = 3) -> List[Dict[str, Any]]:
    if not nodes:
        return []

    suggestions: List[Dict[str, Any]] = []
    signatures = set()

    def add(label: str, new_nodes: List[Dict[str, Any]], rationale: str) -> None:
        signature = tuple((n.get("id"), n.get("title"), n.get("concept")) for n in new_nodes)
        if signature in signatures:
            return
        signatures.add(signature)
        suggestions.append({
            "label": label,
            "nodes": new_nodes,
            "rationale": rationale,
        })

    try:
        by_priority = sorted(nodes, key=lambda n: float(n.get("priority", 0) or 0), reverse=True)
    except Exception:
        by_priority = nodes
    if by_priority != nodes:
        add(
            "Priority-first ordering",
            by_priority,
            "Front-loads higher-priority concepts to accelerate mastery of critical topics.",
        )

    try:
        by_time = sorted(nodes, key=lambda n: float(n.get("time_estimate", 0) or 0))
    except Exception:
        by_time = nodes
    if by_time != nodes:
        add(
            "Time-optimized ordering",
            by_time,
            "Starts with shorter activities to build momentum before longer sessions.",
        )

    if len(nodes) > 1:
        try:
            longest = max(nodes, key=lambda n: float(n.get("time_estimate", 0) or 0))
        except Exception:
            longest = nodes[-1]
        reduced = [n for n in nodes if n is not longest]
        if reduced != nodes:
            removed_title = longest.get("title") or longest.get("name") or longest.get("id")
            add(
                "Compressed workload",
                reduced,
                f"Removes {removed_title} to respect tighter time budgets while keeping sequencing intact.",
            )

    return suggestions[:limit]


def generate_explanation(
    path: Any,
    learner_id: str,
    learner: Optional[Any] = None,
) -> Dict[str, Any]:
    """Create a natural-language explanation for a learning path using an LLM when available."""
    nodes, metrics = _normalize_path(path)
    heuristics = explain_path(nodes, metrics=metrics, learner=learner)
    node_summary = _summarize_nodes(nodes)

    prompt = (
        "You are an educational coach. Given the learner's current plan, craft a concise explanation "
        "(2 short paragraphs) describing why the sequence suits the learner and how it balances priority, "
        "difficulty, and time. Provide a closing sentence with guidance for the learner.\n\n"
        f"Learner ID: {learner_id}\n"
        f"Heuristic summary: {heuristics.get('summary', '')}\n"
        "Path nodes with attributes:\n"
        f"{node_summary}\n"
        "Emphasize educational reasoning, avoid repeating numerical lists verbatim, and write in natural English."
    )

    llm_text = _call_llm(prompt)
    fallback_used = False
    if not llm_text:
        fallback_used = True
        llm_text = heuristics.get("summary", "") + "\n" + "\n".join(f"- {step}" for step in heuristics.get("steps", []))

    return {
        "learner_id": learner_id,
        "explanation": llm_text.strip(),
        "steps": heuristics.get("steps", []),
        "summary": heuristics.get("summary", ""),
        "metrics": heuristics.get("metrics", metrics),
        "caveats": heuristics.get("caveats", []),
        "model": _LLM_MODEL_NAME,
        "fallback": fallback_used,
    }


def generate_counterfactuals(path: Any, max_alternatives: int = 3) -> Dict[str, Any]:
    """Suggest alternative learning paths, using an LLM when present and heuristics as backup."""
    nodes, metrics = _normalize_path(path)
    heuristics = _heuristic_alternatives(nodes, max_alternatives)
    node_summary = _summarize_nodes(nodes)

    prompt = (
        "You are an educational planning assistant. The learner currently follows the path below. "
        "Suggest up to {count} alternative path variants. Each alternative should include a short title, "
        "a description of key changes (swaps, reorderings, additions/removals), and the expected benefit "
        "(e.g., faster completion, remediation focus). Respond as bullet points.".format(count=max_alternatives)
        + "\n\nCurrent path:\n"
        + node_summary
        + "\nHeuristic summary: "
        + (metrics.get("summary") if isinstance(metrics, dict) and "summary" in metrics else "N/A")
    )

    llm_text = _call_llm(prompt)
    if not llm_text:
        lines = [f"- {alt['label']}: {alt['rationale']}" for alt in heuristics]
        llm_text = "Possible adjustments:\n" + "\n".join(lines)
        fallback_used = True
    else:
        fallback_used = False

    return {
        "alternatives_text": llm_text.strip(),
        "heuristic_alternatives": heuristics,
        "model": _LLM_MODEL_NAME,
        "fallback": fallback_used,
    }


def llm_complete(prompt: str) -> Optional[str]:
    """Public helper to run a prompt through the configured LLM with fallback handling."""
    return _call_llm(prompt)


def llm_model_name() -> str:
    """Return the identifier of the currently configured LLM model (or 'heuristic')."""
    _get_llm_client()
    return _LLM_MODEL_NAME
