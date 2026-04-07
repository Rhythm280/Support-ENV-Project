"""
inference.py — Official Baseline Agent for SupportEnv.

MANDATORY REQUIREMENTS:
- Uses OpenAI Client for all LLM calls.
- Reads API_BASE_URL, MODEL_NAME, and HF_TOKEN / API_KEY from env vars.
- STRICT logging format: [START], [STEP n], [END]
- Runs all 3 tasks and computes final composite score.
"""

import os
import sys
import json
import logging
import argparse
import time
from typing import Any, Dict, List, Optional

try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False
    OpenAI = None  # type: ignore
from src.environment import SupportEnv
from src.models import Action, ActionType, TaskName

# ── Config ──────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)

if _OPENAI_AVAILABLE:
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN if HF_TOKEN else "n/a",
    )
else:
    client = None  # type: ignore
    logger.warning("⚠️  'openai' package not installed. LLM mode disabled. Use --agent rule.")

# ── Prompts ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an AI Customer Support Agent evaluating helpdesk tickets.

For each ticket you MUST:
1. CLASSIFY it: billing | technical | general | complaint | positive
2. PRIORITIZE it: low | medium | high
3. RESPOND with a resolution that includes relevant keywords
4. ESCALATE only if the urgency_hint says "urgent" AND tone is "angry"
5. CLOSE it once all steps are done

OUTPUT FORMAT — respond with exactly one JSON action:
{
  "action_type": "classify | prioritize | respond | escalate | close",
  "ticket_id": <integer>,
  "content": "<string or null>"
}"""

FEW_SHOT = """
Example 1 – Classification:
Ticket: "I was charged twice." → {"action_type":"classify","ticket_id":1,"content":"billing"}

Example 2 – Response:
Ticket: "Login broken." → {"action_type":"respond","ticket_id":2,"content":"We have reset your credentials and fixed the login issue. Please retry."}
"""


# ── Agent ────────────────────────────────────────────────────────────────────

class BaselineAgent:
    def __init__(self, mode: str = "llm"):
        self.mode = mode

    def act(self, obs: Any, config: Dict[str, Any]) -> Action:
        if self.mode == "rule":
            return self._rule_act(obs, config)
        return self._llm_act(obs, config)

    def _llm_act(self, obs: Any, config: Dict[str, Any]) -> Action:
        """Call the OpenAI client to decide the next action."""
        if not _OPENAI_AVAILABLE or client is None:
            logger.warning("⚠️  openai package not installed — falling back to rule agent.")
            return self._rule_act(obs, config)
        if not HF_TOKEN or HF_TOKEN == "n/a":
            logger.warning("⚠️  HF_TOKEN not set — falling back to rule-based agent.")
            return self._rule_act(obs, config)

        tickets_str = json.dumps(obs.tickets, indent=2)
        allowed = config.get("allowed_actions", [])
        prompt = (
            f"{FEW_SHOT}\n\nCurrent Tickets:\n{tickets_str}\n\n"
            f"Allowed Actions: {allowed}\n\nDecide the NEXT single action. Output only JSON."
        )

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=150,
            )
            raw = response.choices[0].message.content or ""
            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0].strip()
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0].strip()
            data = json.loads(raw)
            return Action(**data)
        except Exception as exc:
            logger.warning("LLM error (%s) — falling back to rule agent.", exc)
            return self._rule_act(obs, config)

    def _rule_act(self, obs: Any, config: Dict[str, Any]) -> Action:
        """Deterministic rule-based agent for reproducible baseline scores."""
        allowed = config.get("allowed_actions", [])

        for t in obs.tickets:
            tid  = t["id"]
            txt  = t["text"].lower()
            stat = t.get("status", "open")
            deps = t.get("dependencies", [])

            # Skip tickets with unresolved dependencies
            if deps:
                dep_states = {
                    dt["id"]: dt.get("status", "open")
                    for dt in obs.tickets
                    if dt["id"] in deps
                }
                if any(s == "open" for s in dep_states.values()):
                    continue

            # Skip already resolved tickets
            if stat in ("closed", "escalated"):
                continue

            # 1. Classify
            if "classify" in allowed and t.get("category") is None:
                cat = _infer_category(txt)
                return Action(action_type=ActionType.CLASSIFY, ticket_id=tid, content=cat)

            # 2. Prioritize
            if "prioritize" in allowed and t.get("priority") is None:
                pri = _infer_priority(txt)
                return Action(action_type=ActionType.PRIORITIZE, ticket_id=tid, content=pri)

            # 3. Escalate or Respond
            if stat == "open":
                urgency = t.get("urgency_hint", "normal")
                persona = t.get("persona", "polite")
                if "escalate" in allowed and urgency == "urgent" and persona == "angry":
                    return Action(action_type=ActionType.ESCALATE, ticket_id=tid)
                if "respond" in allowed:
                    response = _build_response(txt, t.get("category"))
                    return Action(action_type=ActionType.RESPOND, ticket_id=tid, content=response)

        # 4. Close any fully-handled open ticket
        for t in obs.tickets:
            if t.get("status") == "open" and "close" in allowed:
                return Action(action_type=ActionType.CLOSE, ticket_id=t["id"])

        # Fallback — find ANY open ticket and attempt to close it, or just return an invalid action on ticket 1 if everything is closed but done hasn't triggered.
        for t in obs.tickets:
            if t.get("status") == "open":
                return Action(action_type=ActionType.CLOSE, ticket_id=t["id"])
                
        return Action(action_type=ActionType.CLOSE, ticket_id=abs(obs.tickets[0]["id"]))


def _infer_category(txt: str) -> str:
    if any(k in txt for k in ["charge", "refund", "invoice", "billing", "payment", "subscription"]):
        return "billing"
    if any(k in txt for k in ["crash", "error", "bug", "api", "login", "password", "slow", "blank"]):
        return "technical"
    if any(k in txt for k in ["shameful", "furious", "unacceptable", "disgusting", "angry", "frustrated", "legal"]):
        return "complaint"
    if any(k in txt for k in ["thank", "great", "fantastic", "awesome", "love", "perfect", "stars"]):
        return "positive"
    return "general"


def _infer_priority(txt: str) -> str:
    if any(k in txt for k in ["legal", "fraud", "urgent", "furious", "deleted", "immediately", "now!!", "outrageous"]):
        return "high"
    if any(k in txt for k in ["broken", "crash", "login", "error", "slow", "twice", "charged"]):
        return "medium"
    return "low"


def _build_response(txt: str, category: Optional[str]) -> str:
    """Build a keyword-rich, policy-compliant response."""
    cat = category or _infer_category(txt)
    responses = {
        "billing":   "We sincerely apologize for the billing issue. We have reviewed your charge and will issue a full refund within 3-5 business days. Your subscription has been updated and an invoice confirmation will be sent.",
        "technical": "We have identified and fixed the bug causing this error. Please reset your cache, try logging in again, and run a hard refresh. Our team has deployed a patch to resolve the API issue.",
        "complaint": "We are truly sorry for your frustrating experience. We deeply understand your frustration and take this seriously. A senior manager will contact you within 24 hours to resolve this and offer appropriate compensation.",
        "general":   "Thank you for reaching out! We are happy to help with your question. Our team is available Monday–Friday 9am–6pm UTC. Please check our help center or add team members via Settings > Team.",
        "positive":  "Thank you so much for your wonderful feedback! We are glad to hear about your experience. We appreciate your kind words and have shared this with the team. We look forward to serving you!",
    }
    return responses.get(cat, "Thank you for contacting support. We will resolve your issue promptly.")


# ── Runner ───────────────────────────────────────────────────────────────────

def run_episode(task: str, agent_mode: str, seed: int = 42) -> float:
    env = SupportEnv(task=task, seed=seed, agent_mode=agent_mode)
    obs = env.reset()
    config = env.state()

    # ── [START] ──────────────────────────────────────────────────────────────
    logger.info("[START] task=%s agent=%s seed=%d episode_id=%s",
                task.upper(), agent_mode, seed, env._episode_id)
    logger.info("[START] tickets=%d max_steps=%d allowed_actions=%s",
                len(obs.tickets), config["max_steps"], config["allowed_actions"])

    agent = BaselineAgent(mode=agent_mode)
    done = False
    step_n = 0
    start_time = time.monotonic()

    while not done:
        step_n += 1
        action = agent.act(obs, config)

        logger.info("[STEP %d] action_type=%s ticket_id=%d content=%s",
                    step_n, action.action_type.value, action.ticket_id,
                    repr(action.content) if action.content else "null")

        obs, reward, done, info = env.step(action)

        logger.info("[STEP %d] reward=%.3f cumulative=%.3f tickets_resolved=%d loops=%d",
                    step_n, reward, obs.cumulative_reward,
                    info.get("tickets_resolved", 0), info.get("loops_detected", 0))

        config = env.state()

    elapsed = time.monotonic() - start_time
    report = env.grade()

    # ── [END] ────────────────────────────────────────────────────────────────
    logger.info("[END] task=%s score=%.4f cumulative_reward=%.3f steps=%d elapsed=%.2fs",
                task.upper(), report.score, obs.cumulative_reward, step_n, elapsed)
    logger.info("[END] breakdown=%s", json.dumps(report.breakdown))
    logger.info("[END] summary=%s", report.summary)

    return report.score


def main():
    parser = argparse.ArgumentParser(description="SupportEnv Baseline Agent")
    parser.add_argument("--task",  type=str, choices=["easy", "medium", "hard", "all"], default="all")
    parser.add_argument("--agent", type=str, choices=["rule", "llm"], default="rule")
    parser.add_argument("--seed",  type=int, default=42)
    args = parser.parse_args()

    tasks = ["easy", "medium", "hard"] if args.task == "all" else [args.task]
    scores: List[float] = []

    logger.info("=" * 60)
    logger.info("  SupportEnv Baseline Evaluation")
    logger.info("  Agent: %s | Seed: %d", args.agent.upper(), args.seed)
    logger.info("=" * 60)

    for t in tasks:
        score = run_episode(t, args.agent, seed=args.seed)
        scores.append(score)
        logger.info("")  # blank line between tasks

    if args.task == "all":
        avg = sum(scores) / len(scores)
        logger.info("=" * 60)
        logger.info("  📊 FINAL COMPOSITE SCORE: %.1f%%", avg * 100)
        logger.info("  Easy: %.1f%%  Medium: %.1f%%  Hard: %.1f%%",
                    scores[0] * 100, scores[1] * 100, scores[2] * 100)
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
