from __future__ import annotations

import logging
import random
from collections import deque
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch  # type: ignore
    from torch import nn  # type: ignore
    import torch.nn.functional as F  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    nn = None
    F = None

from .config import Config
from .learner_state import LearnerState
from .data_loader import execute_cypher_query


logger = logging.getLogger(__name__)


RL_MAX_ACTIONS = 10
RL_STATE_BASE_FEATURES = 3
RL_DEFAULT_EPISODES = 25
RL_MAX_STEPS = 6
RL_DEFAULT_TIME_BUDGET = 120.0


class RLLearningPathEnv:
    """Lightweight gym-like environment for learning-path reinforcement learning."""

    def __init__(
        self,
        learner_state: LearnerState,
        candidates: List[Dict[str, Any]],
        time_budget: float,
        max_steps: int,
        max_actions: int = RL_MAX_ACTIONS,
        motivation: float = 0.6,
    ) -> None:
        self.learner_state = learner_state
        self.candidates = candidates[:max_actions]
        self.max_actions = max_actions
        self.action_dim = len(self.candidates)
        self.max_steps = max_steps
        self.total_time_budget = max(30.0, float(time_budget))
        self.motivation = float(np.clip(motivation, 0.0, 1.0))
        self.time_scale = max(15.0, max((float(c.get("time_estimate", 30.0)) for c in self.candidates), default=30.0))
        self.state_dim = RL_STATE_BASE_FEATURES + self.max_actions * 2
        self._initial_mastery = dict(learner_state.mastery)
        self.reset()

    def reset(self) -> np.ndarray:
        self.current_mastery = dict(self._initial_mastery)
        self.time_remaining = float(self.total_time_budget)
        self.steps_taken = 0
        self.history: List[str] = []
        self.visited: set[str] = set()
        return self._build_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        if self.action_dim == 0 or action < 0 or action >= self.action_dim:
            return self._build_state(), -1.0, True, {"selected_node": None, "reason": "invalid"}

        candidate = self.candidates[action]
        node_id = candidate["node_id"]
        time_cost = float(candidate.get("time_estimate", 30.0))
        baseline_mastery = float(self.current_mastery.get(node_id, self.learner_state.get_mastery(node_id, 0.0)))
        deficiency = float(np.clip(1.0 - baseline_mastery, 0.0, 1.0))
        revisit_penalty = 0.2 if node_id in self.visited else 0.0

        improvement = deficiency * 0.5
        self.current_mastery[node_id] = float(np.clip(baseline_mastery + improvement, 0.0, 1.0))
        self.time_remaining -= time_cost
        self.steps_taken += 1
        self.visited.add(node_id)
        self.history.append(node_id)

        time_penalty = max(0.0, time_cost / self.total_time_budget)
        overtime_penalty = max(0.0, -self.time_remaining / self.total_time_budget)
        reward = improvement * 1.5 - time_penalty - revisit_penalty - overtime_penalty

        done = self.steps_taken >= self.max_steps or self.time_remaining <= 0.0
        next_state = self._build_state()
        info = {
            "selected_node": node_id,
            "improvement": improvement,
            "time_cost": time_cost,
            "reward": reward,
        }
        return next_state, float(reward), done, info

    def _build_state(self) -> np.ndarray:
        mastery_values = list(self.current_mastery.values())
        avg_mastery = float(np.clip(np.mean(mastery_values) if mastery_values else 0.0, 0.0, 1.0))
        time_ratio = float(np.clip(self.time_remaining / self.total_time_budget, 0.0, 1.0))
        features: List[float] = [avg_mastery, time_ratio, self.motivation]

        deficiencies: List[float] = []
        time_features: List[float] = []
        for candidate in self.candidates:
            node_id = candidate["node_id"]
            mastery = float(self.current_mastery.get(node_id, self.learner_state.get_mastery(node_id, 0.0)))
            deficiencies.append(float(np.clip(1.0 - mastery, 0.0, 1.0)))
            time_cost = float(candidate.get("time_estimate", 30.0))
            time_features.append(float(np.clip(time_cost / self.time_scale, 0.0, 1.0)))

        while len(deficiencies) < self.max_actions:
            deficiencies.append(0.0)
            time_features.append(0.0)

        state = np.array(features + deficiencies[: self.max_actions] + time_features[: self.max_actions], dtype=np.float32)
        return state

    def get_history(self) -> List[str]:
        return list(self.history)


if nn is not None:

    class _DQNNetwork(nn.Module):
        def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[name-defined]
            return self.layers(x)


class DQNAgent:
    """Minimal DQN agent for curriculum planning."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        device: str,
        learning_rate: float = 1e-3,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.98,
        batch_size: int = 32,
        memory_size: int = 4096,
        target_update: int = 10,
    ) -> None:
        if torch is None or nn is None:
            raise ImportError("DQNAgent requires PyTorch. Please install torch for RL planning.")

        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model = _DQNNetwork(state_dim, action_dim).to(self.device)  # type: ignore[arg-type]
        self.target_model = _DQNNetwork(state_dim, action_dim).to(self.device)  # type: ignore[arg-type]
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.memory: deque = deque(maxlen=memory_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.learn_steps = 0
        self.target_update = target_update

    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state: np.ndarray) -> int:
        if self.action_dim == 0:
            return 0
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())

    def replay(self) -> None:
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states = torch.tensor(np.stack([m[0] for m in minibatch]), dtype=torch.float32, device=self.device)
        actions = torch.tensor([m[1] for m in minibatch], dtype=torch.long, device=self.device)
        rewards = torch.tensor([m[2] for m in minibatch], dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.stack([m[3] for m in minibatch]), dtype=torch.float32, device=self.device)
        dones = torch.tensor([1.0 if m[4] else 0.0 for m in minibatch], dtype=torch.float32, device=self.device)

        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0]
        targets = rewards + (1.0 - dones) * self.gamma * next_q_values

        loss = F.smooth_l1_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        self.learn_steps += 1
        if self.learn_steps % self.target_update == 0:
            self.update_target()

    def update_target(self) -> None:
        self.target_model.load_state_dict(self.model.state_dict())

    def decay_epsilon(self) -> None:
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            if self.epsilon < self.epsilon_min:
                self.epsilon = self.epsilon_min

try:
    # Optional reuse of existing A* if available
    from .path_generator import (
        a_star_custom,
        determine_start_node,
        determine_goal_node,
        calculate_dynamic_weights,
    )
except Exception:  # pragma: no cover - optional integration
    a_star_custom = None  # type: ignore
    determine_start_node = None  # type: ignore
    determine_goal_node = None  # type: ignore
    calculate_dynamic_weights = None  # type: ignore


class AdaptivePathPlanner:
    """
    High-level adapter for personalized path planning.

    This wrapper optionally delegates to the existing A* if present, while
    allowing injection of learner-centric dynamic weights and constraints.
    The goal is to keep this non-breaking and easily swappable.
    """

    def __init__(self, neo4j_driver: Any | None = None) -> None:
        self.driver = neo4j_driver
        if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():  # pragma: no branch
            self.rl_device = "cuda"
        else:
            self.rl_device = "cpu"

    # --- Reinforcement learning helpers ------------------------------------
    def _infer_motivation(self, learner: LearnerState) -> float:
        raw = (
            learner.preferences.get("motivation")
            or learner.preferences.get("motivation_level")
            or learner.preferences.get("engagement_score")
            or 0.6
        )
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return 0.6
        if value > 1.0:
            value /= 100.0
        return float(np.clip(value, 0.1, 1.0))

    def _fetch_candidate_concepts(self, learner_id: str, limit: int = RL_MAX_ACTIONS) -> List[Dict[str, Any]]:
        if not self.driver:
            return []

        query = """
        MATCH (n:KnowledgeNode)
        OPTIONAL MATCH (s:Student {StudentID: $sid})-[:HAS_LEARNING_DATA]->(ld:LearningData)-[:RELATED_TO_NODE]->(n)
        WITH n, count(ld) AS interactions
        RETURN n.Node_ID AS node_id,
               coalesce(toFloat(n.Time_Estimate), 30.0) AS time_estimate,
               coalesce(n.Priority, 0.0) AS priority,
               coalesce(n.Difficulty, 'STANDARD') AS difficulty,
               interactions
        ORDER BY interactions DESC, priority DESC
        LIMIT toInteger($limit)
        """

        try:
            rows = execute_cypher_query(self.driver, query, {"sid": learner_id, "limit": limit})
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("RL candidate retrieval failed for %s: %s", learner_id, exc)
            rows = []

        candidates: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for row in rows or []:
            node_id = str(row.get("node_id") or "").strip()
            if not node_id or node_id in seen:
                continue
            candidate = {
                "node_id": node_id,
                "time_estimate": float(row.get("time_estimate") or 30.0),
                "priority": float(row.get("priority") or 0.0),
                "difficulty": row.get("difficulty") or "STANDARD",
                "interactions": int(row.get("interactions") or 0),
            }
            candidates.append(candidate)
            seen.add(node_id)
            if len(candidates) >= limit:
                break
        return candidates

    def plan_path_with_rl(
        self,
        learner_id: str,
        learner_state: Optional[LearnerState] = None,
        episodes: int = RL_DEFAULT_EPISODES,
        max_steps: int = RL_MAX_STEPS,
    ) -> Dict[str, Any]:
        if torch is None or nn is None or F is None:
            raise ImportError("plan_path_with_rl requires PyTorch. Please install torch to use RL mode.")

        if learner_state is None:
            if not self.driver:
                raise ValueError("Neo4j driver is required to hydrate learner state for RL planning.")
            learner_state = LearnerState.from_neo4j(self.driver, learner_id)

        candidates = self._fetch_candidate_concepts(learner_id, RL_MAX_ACTIONS)
        if not candidates:
            for node_id in list(learner_state.mastery.keys())[:RL_MAX_ACTIONS]:
                candidates.append(
                    {
                        "node_id": node_id,
                        "time_estimate": 30.0,
                        "priority": 0.0,
                        "difficulty": "STANDARD",
                        "interactions": 0,
                    }
                )

        if not candidates:
            return {
                "nodes": [],
                "total_cost": None,
                "meta": {"mode": "rl", "note": "No candidate concepts available for RL planning."},
            }

        time_budget = float(learner_state.time_budget_minutes or RL_DEFAULT_TIME_BUDGET)
        motivation = self._infer_motivation(learner_state)
        env = RLLearningPathEnv(
            learner_state=learner_state,
            candidates=candidates,
            time_budget=time_budget,
            max_steps=max_steps,
            motivation=motivation,
        )

        if env.action_dim == 0:
            return {
                "nodes": [],
                "total_cost": None,
                "meta": {"mode": "rl", "note": "RL environment has no available actions."},
            }

        agent = DQNAgent(env.state_dim, env.action_dim, self.rl_device)
        agent.model.train()  # type: ignore[operator]

        best_reward = float("-inf")
        best_history: List[str] = []

        for _ in range(max(1, episodes)):
            state = env.reset()
            total_reward = 0.0
            done = False
            while not done:
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                agent.replay()
                state = next_state
                total_reward += reward
            agent.decay_epsilon()
            history = env.get_history()
            if total_reward > best_reward and history:
                best_reward = total_reward
                best_history = history.copy()

        agent.update_target()
        agent.model.eval()  # type: ignore[operator]

        eval_env = RLLearningPathEnv(
            learner_state=learner_state,
            candidates=candidates,
            time_budget=time_budget,
            max_steps=max_steps,
            motivation=motivation,
        )
        eval_state = eval_env.reset()
        eval_reward = 0.0
        eval_history: List[str] = []
        epsilon_backup = agent.epsilon
        agent.epsilon = 0.0

        done = False
        while not done:
            action = agent.act(eval_state)
            next_state, reward, done, info = eval_env.step(action)
            eval_state = next_state
            eval_reward += reward
            selected = info.get("selected_node")
            if selected:
                eval_history.append(selected)

        agent.epsilon = epsilon_backup

        if not eval_history and best_history:
            eval_history = best_history
            eval_reward = best_reward

        unique_path: List[str] = []
        seen_nodes: set[str] = set()
        for node in eval_history:
            if node not in seen_nodes:
                unique_path.append(node)
                seen_nodes.add(node)

        candidate_lookup = {c["node_id"]: c for c in candidates}
        detailed_nodes = [candidate_lookup[n] for n in unique_path if n in candidate_lookup]

        return {
            "nodes": unique_path,
            "total_cost": None,
            "meta": {
                "mode": "rl",
                "episodes": episodes,
                "training_best_reward": best_reward,
                "evaluation_reward": eval_reward,
                "epsilon_final": agent.epsilon,
                "candidates": detailed_nodes,
            },
        }

    # --- Dynamic weight logic -------------------------------------------------
    def compute_dynamic_weights(self, learner: LearnerState) -> Dict[str, float]:
        """
        Convert LearnerState into weight scalars compatible with heuristic terms.
        This mirrors calculate_dynamic_weights but allows additional signals.
        """
        # Time scalar: slower learners value time penalty more; if user set pace
        pace = learner.preferences.get("pace", "normal")
        base_time = 1.0
        if pace == "fast":
            base_time = 0.75
        elif pace == "slow":
            base_time = 1.25

        # Optional mastery-driven emphasis: if low mastery overall, reduce time weight
        if learner.mastery:
            avg_mastery = sum(learner.mastery.values()) / max(1, len(learner.mastery))
        else:
            avg_mastery = 0.0
        mastery_factor = 0.9 + (1.0 - avg_mastery) * 0.2  # 0.9..1.1

        dynamic = {
            "difficulty_standard": 1.0,
            "difficulty_advanced": 1.0,
            "skill_level": 1.0,
            "time_estimate": base_time * mastery_factor,
        }
        return dynamic

    def _infer_low_mastery_node(self, learner: LearnerState, threshold: float = 0.5) -> Optional[str]:
        if not learner.mastery:
            return None
        ordered = sorted(learner.mastery.items(), key=lambda item: item[1])
        for node_id, mastery in ordered:
            if mastery < threshold:
                return node_id
        return ordered[0][0] if ordered else None

    def _fetch_time_estimates(self, node_ids: List[str]) -> Dict[str, float]:
        if not node_ids or not self.driver:
            return {}
        query = """
        MATCH (n:KnowledgeNode)
        WHERE n.Node_ID IN $ids
        RETURN n.Node_ID AS node_id, coalesce(toFloat(n.Time_Estimate), 30.0) AS time_estimate
        """
        try:
            rows = execute_cypher_query(self.driver, query, {"ids": node_ids})
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.debug("Time estimate lookup failed: %s", exc)
            return {}
        return {str(row.get("node_id")): float(row.get("time_estimate") or 30.0) for row in rows or []}

    # --- Planning API ---------------------------------------------------------
    def plan_path(
        self,
        learner: LearnerState,
        start_concept: Optional[str] = None,
        goal_concept: Optional[str] = None,
        mode: str = "a_star",
        max_depth: int = 50,
        hard_time_budget_minutes: Optional[int] = None,
        extra_constraints: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Produce a personalized path plan via A* (default) or RL when mode='rl'.

        Returns a dict including: nodes, total_cost, meta.
        """
        mode_normalized = (mode or "a_star").lower()

        if mode_normalized == "rl":
            rl_episodes = RL_DEFAULT_EPISODES
            rl_max_steps = min(max_depth, RL_MAX_STEPS)
            if extra_constraints:
                raw_episodes = extra_constraints.get("rl_episodes")
                if raw_episodes is not None:
                    try:
                        rl_episodes = max(1, int(raw_episodes))
                    except (TypeError, ValueError):  # pragma: no cover - defensive
                        logger.debug("Ignoring invalid rl_episodes constraint: %s", raw_episodes)
                raw_steps = extra_constraints.get("rl_max_steps")
                if raw_steps is not None:
                    try:
                        rl_max_steps = max(1, int(raw_steps))
                    except (TypeError, ValueError):  # pragma: no cover - defensive
                        logger.debug("Ignoring invalid rl_max_steps constraint: %s", raw_steps)

            original_budget = None
            if hard_time_budget_minutes is not None:
                original_budget = learner.time_budget_minutes
                learner.time_budget_minutes = hard_time_budget_minutes
            try:
                return self.plan_path_with_rl(
                    learner_id=learner.student_id,
                    learner_state=learner,
                    episodes=rl_episodes,
                    max_steps=rl_max_steps,
                )
            finally:
                if original_budget is not None:
                    learner.time_budget_minutes = original_budget

        if a_star_custom is None:
            # Skeleton fallback when A* is unavailable
            return {
                "nodes": [],
                "total_cost": 0.0,
                "meta": {
                    "note": "A* implementation not available; returning empty plan.",
                },
            }

        # Determine dynamic weights; optionally combine with existing function
        dyn = self.compute_dynamic_weights(learner)
        if calculate_dynamic_weights is not None:
            try:
                existing_dyn = calculate_dynamic_weights(learner.student_id)
                # Merge with precedence to our computed values when set
                dyn = {**existing_dyn, **dyn}
            except Exception:
                pass

        # Determine start/goal when not provided, if helpers exist
        if start_concept is None and determine_start_node is not None:
            try:
                start_concept = determine_start_node(learner.student_id)
            except Exception:
                start_concept = None
        if goal_concept is None and determine_goal_node is not None:
            try:
                goal_concept = determine_goal_node(learner.student_id)
            except Exception:
                goal_concept = None

        # Guard — cannot proceed without nodes
        if not start_concept or not goal_concept:
            return {
                "nodes": [],
                "total_cost": 0.0,
                "meta": {
                    "note": "Start or goal concept missing; returning empty plan.",
                },
            }

        # Delegate to existing A*
        try:
            result = a_star_custom(
                start_concept,
                goal_concept,
                dynamic_weights=dyn,
                max_depth=max_depth,
                time_budget_minutes=hard_time_budget_minutes,
                extra_constraints=extra_constraints or {},
            )
        except TypeError:
            # Older signature that doesn't support new params
            result = a_star_custom(
                start_concept,
                goal_concept,
                dynamic_weights=dyn,
            )

        return {
            "nodes": result.get("path", []),
            "total_cost": result.get("total_cost", 0.0),
            "meta": {
                "expanded": result.get("expanded", 0),
                "visited": result.get("visited", 0),
                "weights": dyn,
            },
        }

    def plan_hybrid_path(
        self,
        learner_id: str,
        goal_concept: Optional[str] = None,
        max_depth: int = 50,
    ) -> Dict[str, Any]:
        if not self.driver:
            raise ValueError("Neo4j driver required for hybrid planning.")

        learner_state = LearnerState.from_neo4j(self.driver, learner_id)

        start_concept = None
        if determine_start_node is not None:
            try:
                start_concept = determine_start_node(learner_id)
            except Exception:
                start_concept = None
        if not start_concept:
            start_concept = self._infer_low_mastery_node(learner_state)

        base_plan = self.plan_path(
            learner=learner_state,
            start_concept=start_concept,
            goal_concept=goal_concept,
            mode="a_star",
            max_depth=max_depth,
            hard_time_budget_minutes=learner_state.time_budget_minutes,
        )
        base_nodes: List[str] = list(base_plan.get("nodes", []) or [])

        base_time_lookup = self._fetch_time_estimates(base_nodes)
        base_estimated_time = sum(base_time_lookup.get(node, 30.0) for node in base_nodes)

        if learner_state.mastery:
            avg_mastery = mean(learner_state.get_mastery(node, 0.0) for node in base_nodes) if base_nodes else mean(learner_state.mastery.values())
        else:
            avg_mastery = None

        time_budget = float(learner_state.time_budget_minutes) if learner_state.time_budget_minutes else None
        time_slack = (time_budget - base_estimated_time) if time_budget is not None else None

        deficiency_nodes = {node for node in base_nodes if learner_state.get_mastery(node, 0.0) < 0.45}
        needs_adjustment = (
            not base_nodes
            or (avg_mastery is not None and avg_mastery < 0.55)
            or (time_slack is not None and time_slack < -15.0)
        )

        rl_result: Optional[Dict[str, Any]] = None
        rl_nodes: List[str] = []
        if needs_adjustment:
            try:
                rl_steps = max(3, min(RL_MAX_STEPS, len(base_nodes) + 2)) if base_nodes else RL_MAX_STEPS
                rl_result = self.plan_path_with_rl(
                    learner_id=learner_id,
                    learner_state=learner_state,
                    episodes=max(10, RL_DEFAULT_EPISODES // 2),
                    max_steps=rl_steps,
                )
                rl_nodes = list(rl_result.get("nodes", []) or [])
            except ImportError:
                logger.warning("PyTorch unavailable; falling back to A* plan only for %s.", learner_id)
                needs_adjustment = False
            except Exception as exc:  # pragma: no cover
                logger.warning("RL planning failed for %s: %s", learner_id, exc)
                needs_adjustment = False

        hybrid_nodes: List[str] = list(base_nodes)
        inserted: set[str] = set()

        if rl_nodes:
            for base_node in base_nodes:
                if base_node in deficiency_nodes:
                    for candidate in rl_nodes:
                        if candidate not in inserted and candidate not in hybrid_nodes:
                            hybrid_nodes.insert(hybrid_nodes.index(base_node), candidate)
                            inserted.add(candidate)
                # ensure base node remains present, index recomputed next iteration
            for candidate in rl_nodes:
                if candidate not in hybrid_nodes:
                    hybrid_nodes.append(candidate)

        hybrid_time_lookup = self._fetch_time_estimates(hybrid_nodes)
        hybrid_estimated_time = sum(hybrid_time_lookup.get(node, 30.0) for node in hybrid_nodes)

        rl_meta = (rl_result or {}).get("meta", {}) if rl_result else {}
        meta = {
            "mode": "hybrid",
            "base_nodes": base_nodes,
            "rl_nodes": rl_nodes,
            "needs_adjustment": needs_adjustment,
            "avg_mastery_base": avg_mastery,
            "estimated_time_base": base_estimated_time,
            "estimated_time_hybrid": hybrid_estimated_time,
            "time_budget": time_budget,
            "time_slack": time_slack,
            "deficiency_nodes": sorted(deficiency_nodes),
            "rl_meta": {k: rl_meta.get(k) for k in ("episodes", "training_best_reward", "evaluation_reward", "epsilon_final") if k in rl_meta},
        }

        total_cost = base_plan.get("total_cost") if base_plan else None
        if total_cost is None:
            total_cost = hybrid_estimated_time if hybrid_nodes else None

        return {
            "nodes": hybrid_nodes,
            "total_cost": total_cost,
            "meta": meta,
        }
