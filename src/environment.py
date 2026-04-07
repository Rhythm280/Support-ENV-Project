"""
environment.py — Core SupportEnv class.

Implements a standard gym-style API:
    env = SupportEnv(task="easy" | "medium" | "hard", seed=42)
    obs  = env.reset()
    obs, reward, done, info = env.step(action)
    state = env.state()         → full state dict (includes ground-truth)
    report = env.grade()        → GradeReport (0.0–1.0)
    actions = env.replay(episode_id)  → reconstructed action sequence

New in this version:
    - Uses generator.py (procedural, seed-based) instead of static ticket bank
    - SQLite persistence via database.py
    - Dynamic state via dynamics.py (priority escalation, tone, dependencies)
    - Dense info dict: tickets_resolved, efficiency, loops_detected, priority_misses
    - Loop detection with -0.5 penalty via rewards.py
    - Optional Supabase analytics via supabase_client.py
    - Episode replay from SQLite
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.models import (
    Action,
    ActionType,
    Observation,
    StepResult,
    TaskName,
    TicketStatus,
    Ticket,
)
from src.rewards import R_INVALID_ACTION, calculate_reward
from src.tasks import TASK_CONFIGS, GradeReport, grade
from src.generator import generate_tickets
from src.dynamics import (
    apply_priority_escalation,
    assign_dependencies,
    check_dependencies_met,
    partial_observation,
    worsen_tone,
)
import src.database as db
from src.supabase_client import sb_log_episode, sb_log_action, sb_log_metrics

logger = logging.getLogger(__name__)


class SupportEnv:
    """
    Customer Support Ticket Resolution RL Environment.

    Parameters
    ----------
    task : str | TaskName
        One of "easy", "medium", "hard".
    seed : int
        Random seed for ticket generation (ensures reproducibility).
    agent_mode : str
        Label for logging purposes ("rule", "llm", "human").
    enable_dynamics : bool
        If True, apply dynamic state rules (priority escalation, tone worsening).
    db_path : str
        Path to SQLite database (default: support_env.db).
    """

    def __init__(
        self,
        task: str | TaskName = TaskName.EASY,
        seed: int = 42,
        agent_mode: str = "unknown",
        enable_dynamics: bool = True,
        db_path: str = "support_env.db",
    ):
        self.task_name = TaskName(task)
        self.config = TASK_CONFIGS[self.task_name]
        self.seed = seed
        self.agent_mode = agent_mode
        self.enable_dynamics = enable_dynamics
        self.db_path = db_path

        # Initialize SQLite DB
        db.init_db(db_path)

        # Internal state
        self._tickets: List[Ticket] = []
        self._step_count = 0
        self._cumulative_reward = 0.0
        self._done = False
        self._step_log: List[Dict] = []
        self._action_history: List[tuple] = []  # (action_type, ticket_id)
        self._bad_response_counts: Dict[int, int] = {}  # ticket_id → count
        self._episode_id: Optional[int] = None
        self._loops_detected = 0
        self._priority_misses = 0

    # ─────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None) -> Observation:
        """Reset the environment and return the initial observation."""
        if seed is not None:
            self.seed = seed

        # Generate tickets procedurally
        self._tickets = generate_tickets(
            count=self.config.ticket_count,
            seed=self.seed,
        )

        # Assign dependencies for Hard task
        if self.task_name == TaskName.HARD:
            assign_dependencies(self._tickets, seed=self.seed)

        # Reset episode state
        self._step_count = 0
        self._cumulative_reward = 0.0
        self._done = False
        self._step_log = []
        self._action_history = []
        self._bad_response_counts = {t.id: 0 for t in self._tickets}
        self._loops_detected = 0
        self._priority_misses = 0

        # Log episode start in SQLite
        self._episode_id = db.log_episode_start(
            task=self.task_name.value,
            seed=self.seed,
            agent_mode=self.agent_mode,
            db_path=self.db_path,
        )

        logger.info(
            "SupportEnv: Reset — task=%s seed=%d tickets=%d episode_id=%s",
            self.task_name, self.seed, len(self._tickets), self._episode_id,
        )
        return self._build_observation(error=False)

    def add_ticket(
        self,
        text: str,
        true_category: str,
        true_priority: str,
        persona: str = "polite",
        requires_escalation: bool = False,
    ) -> Ticket:
        """Inject a custom ticket into the environment state."""
        from src.models import Ticket, Category, Priority
        
        # Determine next ID
        next_id = max([t.id for t in self._tickets], default=0) + 1
        
        ticket = Ticket(
            id=next_id,
            text=text,
            true_category=Category(true_category),
            true_priority=Priority(true_priority),
            persona=persona,
            requires_escalation=requires_escalation,
        )
        
        self._tickets.append(ticket)
        self._bad_response_counts[ticket.id] = 0
        
        logger.info("SupportEnv: Injected custom ticket #%d: %s...", ticket.id, text[:30])
        return ticket

    def step(self, action: Action) -> tuple[Observation, float, bool, dict]:
        """
        Execute one action and return (observation, reward, done, info).

        Info dict includes:
            tickets_resolved: int
            efficiency: float (resolved / step_count)
            loops_detected: int
            priority_misses: int
        """
        if self._done:
            raise RuntimeError("Environment is done. Call reset() before stepping.")

        self._step_count += 1
        reward = 0.0
        error = False
        reason = ""

        # ── Apply dynamic state rules ─────────────────────────────────────────
        if self.enable_dynamics:
            apply_priority_escalation(self._tickets, self._step_count)

        # ── Find ticket ───────────────────────────────────────────────────────
        ticket = next((t for t in self._tickets if t.id == action.ticket_id), None)
        if ticket is None:
            reward = R_INVALID_ACTION
            error = True
            reason = f"Ticket ID {action.ticket_id} not found"
            return self._finalize_step(reward, error, reason)

        # ── Action gating ─────────────────────────────────────────────────────
        if action.action_type not in self.config.allowed_actions:
            reward = R_INVALID_ACTION
            error = True
            reason = f"Action '{action.action_type}' not allowed in task '{self.task_name}'"
            return self._finalize_step(reward, error, reason)

        # ── Dependency check (Hard task) ──────────────────────────────────────
        if self.task_name == TaskName.HARD and ticket.dependencies:
            if not check_dependencies_met(ticket, self._tickets):
                reward = R_INVALID_ACTION
                error = True
                reason = f"Ticket #{ticket.id} blocked by dependencies {ticket.dependencies}"
                return self._finalize_step(reward, error, reason)

        # ── Calculate reward (with history for loop detection) ────────────────
        reward, reason = calculate_reward(ticket, action, self._action_history)

        # Track loop detection
        if "loop" in reason.lower() or "repeated" in reason.lower():
            self._loops_detected += 1

        # ── Apply action to ticket state ───────────────────────────────────────
        self._apply_action(ticket, action)

        # ── Tone worsening for bad responses ──────────────────────────────────
        if self.enable_dynamics and action.action_type == ActionType.RESPOND:
            if reward < 0:
                self._bad_response_counts[ticket.id] = (
                    self._bad_response_counts.get(ticket.id, 0) + 1
                )
                worsen_tone(ticket, self._bad_response_counts[ticket.id])

        # ── Track priority misses ─────────────────────────────────────────────
        if (action.action_type == ActionType.PRIORITIZE and
                ticket.priority != ticket.true_priority):
            self._priority_misses += 1

        # ── Update cumulative reward ───────────────────────────────────────────
        self._cumulative_reward += reward

        logger.debug(
            "[step %d] action=%s ticket=%d reward=%.2f reason=%s",
            self._step_count, action.action_type, action.ticket_id, reward, reason,
        )

        # ── Record action history ─────────────────────────────────────────────
        self._action_history.append((action.action_type.value, action.ticket_id))

        # ── Persist step to SQLite ────────────────────────────────────────────
        if self._episode_id is not None:
            db.log_action(
                episode_id=self._episode_id,
                step=self._step_count,
                action_type=action.action_type.value,
                ticket_id=action.ticket_id,
                content=action.content,
                reward=reward,
                cumulative_reward=self._cumulative_reward,
                reason=reason,
                db_path=self.db_path,
            )
            sb_log_action(
                self._episode_id, self._step_count,
                action.action_type.value, action.ticket_id, reward, reason,
            )

        self._step_log.append({
            "step": self._step_count,
            "action": action.action_type.value,
            "ticket_id": action.ticket_id,
            "content": action.content,
            "reward": round(reward, 3),
            "reason": reason,
            "cumulative": round(self._cumulative_reward, 3),
        })

        return self._finalize_step(reward, error, reason)

    def _check_done(self) -> bool:
        """Centralized done check evaluated on every return."""
        if self._step_count >= self.config.max_steps:
            return True
        all_resolved = all(
            t.status in (TicketStatus.CLOSED, TicketStatus.ESCALATED)
            for t in self._tickets
        )
        return all_resolved

    def state(self) -> Dict[str, Any]:
        """
        Full environment state including ground-truth labels.
        Intended for the dashboard and graders — NOT for agent consumption.
        """
        return {
            "task": self.task_name.value,
            "seed": self.seed,
            "episode_id": self._episode_id,
            "step_count": self._step_count,
            "max_steps": self.config.max_steps,
            "cumulative_reward": round(self._cumulative_reward, 3),
            "done": self._done,
            "tickets": [t.model_dump() for t in self._tickets],
            "step_log": self._step_log,
            "allowed_actions": [a.value for a in self.config.allowed_actions],
            "info": self._compute_info(),
        }

    def grade(self) -> GradeReport:
        """Score the current episode using the task-specific grader."""
        report = grade(self.task_name, self._tickets)

        # Persist final state to SQLite
        if self._episode_id is not None:
            db.log_episode_end(
                episode_id=self._episode_id,
                total_steps=self._step_count,
                final_score=report.score,
                cumulative_reward=self._cumulative_reward,
                done=self._done,
                db_path=self.db_path,
            )
            db.log_tickets(self._episode_id, self._tickets, db_path=self.db_path)
            info = self._compute_info()
            db.log_metrics(
                episode_id=self._episode_id,
                tickets_resolved=info["tickets_resolved"],
                efficiency=info["efficiency"],
                loops_detected=info["loops_detected"],
                priority_misses=info["priority_misses"],
                final_score=report.score,
                breakdown=report.breakdown,
                db_path=self.db_path,
            )
            sb_log_episode(
                self._episode_id, self.task_name.value, self.seed,
                report.score, self._cumulative_reward, self._step_count,
                self.agent_mode,
            )
            sb_log_metrics(self._episode_id, {"final_score": report.score, **report.breakdown})

        return report

    def replay(self, episode_id: int) -> List[Dict]:
        """
        Replay a past episode from the SQLite database.

        Loads the episode action sequence and returns it for inspection.
        To re-execute, call env.reset() with the original seed, then
        feed each action dict back through env.step().
        """
        actions = db.replay_episode(episode_id, db_path=self.db_path)
        episode_data = db.load_episode(episode_id, db_path=self.db_path)
        logger.info(
            "SupportEnv: Replaying episode %d | task=%s seed=%d steps=%d score=%.3f",
            episode_id,
            episode_data["episode"]["task"],
            episode_data["episode"]["seed"],
            episode_data["episode"]["total_steps"],
            episode_data["episode"]["final_score"],
        )
        return actions

    # ─────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────

    def _apply_action(self, ticket: Ticket, action: Action) -> None:
        """Mutate ticket state based on the action taken."""
        if action.action_type == ActionType.CLASSIFY:
            from src.models import Category
            try:
                ticket.category = Category(action.content)
            except (ValueError, TypeError):
                pass

        elif action.action_type == ActionType.PRIORITIZE:
            from src.models import Priority
            try:
                ticket.priority = Priority(action.content)
            except (ValueError, TypeError):
                pass

        elif action.action_type == ActionType.RESPOND:
            ticket.response = action.content
            ticket.status = TicketStatus.CLOSED

        elif action.action_type == ActionType.ESCALATE:
            ticket.status = TicketStatus.ESCALATED

        elif action.action_type == ActionType.CLOSE:
            ticket.status = TicketStatus.CLOSED

    def _compute_info(self) -> Dict[str, Any]:
        """Compute dense metrics dict for step() and state()."""
        tickets_resolved = sum(
            1 for t in self._tickets
            if t.status in (TicketStatus.CLOSED, TicketStatus.ESCALATED)
        )
        efficiency = (
            tickets_resolved / self._step_count
            if self._step_count > 0 else 0.0
        )
        return {
            "tickets_resolved": tickets_resolved,
            "efficiency": round(efficiency, 3),
            "loops_detected": self._loops_detected,
            "priority_misses": self._priority_misses,
        }

    def _build_observation(self, error: bool) -> Observation:
        return Observation(
            tickets=[partial_observation(t) for t in self._tickets],
            step_count=self._step_count,
            cumulative_reward=round(self._cumulative_reward, 3),
            last_action_error=error,
            task=self.task_name.value,
            info=self._compute_info(),
        )

    def _finalize_step(
        self,
        reward: float,
        error: bool,
        reason: str,
    ) -> tuple[Observation, float, bool, dict]:
        done = self._check_done()
        self._done = done
        
        obs = self._build_observation(error)
        info = {**self._compute_info(), "reason": reason, "step": self._step_count}
        return obs, round(reward, 3), done, info

    def _build_result(
        self,
        reward: float,
        error: bool,
        reason: str,
        done: bool = False,
    ) -> StepResult:
        """Deprecated: for API compat. Use _finalize_step for core logic."""
        obs, r, d, info = self._finalize_step(reward, error, reason, done)
        return StepResult(observation=obs, reward=r, done=d, info=info)
