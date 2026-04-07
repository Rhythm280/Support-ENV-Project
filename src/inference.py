"""
inference.py — Baseline agent runner.

Two agents are provided:
  1. RuleBasedAgent — pure keyword matching, no ML required
  2. LLMAgent       — GPT-powered (gracefully falls back to rule-based if no API key)

Usage:
    python -m src.inference --task easy   --agent rule
    python -m src.inference --task medium --agent rule
    python -m src.inference --task hard   --agent rule
    python -m src.inference --task hard   --agent llm   [requires OPENAI_API_KEY]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Optional

from src.environment import SupportEnv
from src.models import Action, ActionType, Category, Priority, TaskName

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Agent base class
# ─────────────────────────────────────────────

class BaseAgent(ABC):
    """Abstract agent interface."""

    @abstractmethod
    def act(self, observation: dict, ticket: dict) -> Optional[Action]:
        """
        Decide on one action for the given ticket.
        Return None if no more actions needed for this ticket.
        """


# ─────────────────────────────────────────────
# Rule-Based Agent
# ─────────────────────────────────────────────

class RuleBasedAgent(BaseAgent):
    """
    Keyword-matching agent — no ML, no API.
    Demonstrates the minimum bar of environmental interactions.
    """

    def act(self, observation: dict, ticket: dict) -> Optional[Action]:
        text  = ticket["text"].lower()
        tid   = ticket["id"]
        task  = observation["task"]
        cat   = ticket.get("category")
        pri   = ticket.get("priority")
        status = ticket.get("status", "open")

        if status != "open":
            return None  # already handled

        # ── Easy: classify ────────────────────────────────────────────
        if cat is None and task in ("easy", "medium", "hard"):
            return Action(
                action_type=ActionType.CLASSIFY,
                ticket_id=tid,
                content=self._guess_category(text),
            )

        # ── Medium / Hard: prioritize ─────────────────────────────────
        if pri is None and task in ("medium", "hard"):
            return Action(
                action_type=ActionType.PRIORITIZE,
                ticket_id=tid,
                content=self._guess_priority(text),
            )

        # ── Hard: respond or escalate ─────────────────────────────────
        if task == "hard":
            if self._should_escalate(text):
                return Action(action_type=ActionType.ESCALATE, ticket_id=tid)
            return Action(
                action_type=ActionType.RESPOND,
                ticket_id=tid,
                content=self._generate_response(text),
            )

        return None  # nothing left to do for this ticket in current task

    # ── Heuristics ────────────────────────────────────────────────────

    def _guess_category(self, text: str) -> str:
        if any(w in text for w in ["refund", "charge", "invoice", "billing", "payment", "subscription", "cancel"]):
            return Category.BILLING
        if any(w in text for w in ["crash", "error", "bug", "slow", "login", "api", "password", "blank", "500"]):
            return Category.TECHNICAL
        if any(w in text for w in ["angry", "unacceptable", "furious", "disgrace", "terrible", "apologize", "disappointed"]):
            return Category.COMPLAINT
        if any(w in text for w in ["thank", "great", "fantastic", "love", "amazing", "excellent", "5 stars"]):
            return Category.POSITIVE
        return Category.GENERAL

    def _guess_priority(self, text: str) -> str:
        if any(w in text for w in ["immediately", "urgent", "critical", "blocking", "legal", "crash", "angry", "furious", "twice", "deleted"]):
            return Priority.HIGH
        if any(w in text for w in ["slow", "intermittent", "sometimes", "occasional"]):
            return Priority.MEDIUM
        return Priority.LOW

    def _should_escalate(self, text: str) -> bool:
        return any(w in text for w in ["furious", "legal action", "disgrace", "ignoring", "unacceptable", "shameful"])

    def _generate_response(self, text: str) -> str:
        if any(w in text for w in ["refund", "charge", "billing", "invoice", "payment"]):
            return (
                "Thank you for contacting us about your billing concern. "
                "We sincerely apologize for any incorrect charge or refund delay. "
                "I've flagged this for our billing team and will process a refund within 3–5 business days."
            )
        if any(w in text for w in ["crash", "error", "bug", "slow", "login", "api", "password", "blank"]):
            return (
                "Thank you for reporting this technical issue. "
                "Our engineering team has been notified and is actively investigating the bug. "
                "We'll provide an update within 24 hours. Please try clearing your cache in the meantime."
            )
        if any(w in text for w in ["angry", "disappointed", "terrible", "disgrace"]):
            return (
                "We deeply apologize for the experience you've had. "
                "Your frustration is completely understandable, and we're prioritizing this complaint. "
                "A senior support manager will reach out to you within 2 hours."
            )
        if any(w in text for w in ["thank", "great", "amazing", "love"]):
            return (
                "Thank you so much for your kind words! "
                "We're thrilled to hear you're having a positive experience. "
                "Your feedback has been shared with the team — it makes a huge difference!"
            )
        return (
            "Thank you for reaching out to our support team. "
            "We've received your message and will respond within 24 hours. "
            "If you need immediate assistance, feel free to call our helpline."
        )


# ─────────────────────────────────────────────
# LLM Agent (graceful degradation)
# ─────────────────────────────────────────────

class LLMAgent(BaseAgent):
    """
    OpenAI-powered agent. Falls back to RuleBasedAgent if no API key is found.
    """

    def __init__(self):
        self._client = None
        self._fallback = RuleBasedAgent()
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=api_key)
                logger.info("LLMAgent: OpenAI client initialized ✓")
            except ImportError:
                logger.warning("openai package not installed — falling back to rule-based agent")
        else:
            logger.warning("OPENAI_API_KEY not set — falling back to rule-based agent")

    def act(self, observation: dict, ticket: dict) -> Optional[Action]:
        if self._client is None:
            return self._fallback.act(observation, ticket)

        status = ticket.get("status", "open")
        if status != "open":
            return None

        task = observation["task"]
        prompt = self._build_prompt(observation, ticket, task)

        try:
            response = self._client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=300,
            )
            raw = response.choices[0].message.content.strip()
            return self._parse_action(raw, ticket["id"])
        except Exception as exc:
            logger.error("LLM call failed: %s — using rule-based fallback", exc)
            return self._fallback.act(observation, ticket)

    def _build_prompt(self, obs: dict, ticket: dict, task: str) -> str:
        allowed = obs.get("allowed_actions", [])
        return f"""You are a customer support AI agent. Your job is to handle a support ticket.

TASK: {task}
ALLOWED ACTIONS: {allowed}

TICKET:
  ID: {ticket['id']}
  Text: "{ticket['text']}"
  Current category: {ticket.get('category') or 'not set'}
  Current priority: {ticket.get('priority') or 'not set'}
  Status: {ticket.get('status', 'open')}

Respond with a single JSON action object:
{{
  "action_type": "<one of {allowed}>",
  "ticket_id": {ticket['id']},
  "content": "<category name | priority name | response text | null>"
}}

Categories: billing, technical, general, complaint, positive
Priorities: low, medium, high

Rules:
- For "classify": content must be a category name
- For "prioritize": content must be a priority name
- For "respond": content must be a helpful, relevant response string
- For "escalate" or "close": content can be null
- Only escalate if ticket is a complaint or very urgent

Return ONLY the JSON, no explanation."""

    def _parse_action(self, raw: str, ticket_id: int) -> Optional[Action]:
        try:
            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            data = json.loads(raw.strip())
            return Action(**data)
        except Exception as exc:
            logger.error("Failed to parse LLM output: %s\nRaw: %s", exc, raw)
            return None


# ─────────────────────────────────────────────
# Episode runner
# ─────────────────────────────────────────────

def run_episode(task: str, agent_type: str, seed: int = 42) -> None:
    agent_cls = LLMAgent if agent_type == "llm" else RuleBasedAgent
    agent = agent_cls()

    env = SupportEnv(task=task, seed=seed)
    obs = env.reset()

    print(f"\n{'='*60}")
    print(f"  Task: {task.upper()} | Agent: {agent_type.upper()} | Seed: {seed}")
    print(f"{'='*60}")
    print(f"  Tickets loaded: {len(obs.tickets)}")
    print(f"  Max steps: {env.config.max_steps}")
    print(f"  Allowed actions: {[a.value for a in env.config.allowed_actions]}")
    print(f"{'='*60}\n")

    step = 0
    while not env._done:
        acted = False
        for ticket in obs.tickets:
            action = agent.act(obs.model_dump(), ticket)
            if action is None:
                continue

            # env.step() returns (obs, reward, done, info) tuple
            obs, reward, done, info = env.step(action)
            print(
                f"  Step {info.get('step', step):>2} | "
                f"ticket={action.ticket_id} | "
                f"action={action.action_type.value:<11}| "
                f"reward={reward:+.2f} | "
                f"cum={obs.cumulative_reward:+.2f}  | "
                f"{info.get('reason', '')}"
            )
            acted = True

            if done or env._done:
                break

        if not acted or env._done:
            break

        step += 1

    report = env.grade()
    # Safety clamp: ensure score is strictly in (0, 1) — never exactly 0.0 or 1.0
    final_score = max(0.01, min(0.99, float(report.score)))
    print(f"\n{'='*60}")
    print(f"  FINAL SCORE: {final_score:.1%}")
    print(f"  {report.summary}")
    print(f"  Breakdown: {report.breakdown}")
    print(f"  Cumulative reward: {env._cumulative_reward:+.3f}")
    print(f"{'='*60}\n")


# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run SupportEnv baseline agent")
    parser.add_argument("--task",  choices=["easy", "medium", "hard"], default="easy")
    parser.add_argument("--agent", choices=["rule", "llm"],            default="rule")
    parser.add_argument("--seed",  type=int,                           default=42)
    args = parser.parse_args()
    run_episode(args.task, args.agent, args.seed)


if __name__ == "__main__":
    main()
