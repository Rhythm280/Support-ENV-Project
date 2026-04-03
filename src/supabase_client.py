"""
supabase_client.py — Optional Supabase analytics integration for SupportEnv.

Controlled by environment variables:
  ENABLE_SUPABASE=true   — Enable Supabase logging
  SUPABASE_URL           — Your Supabase project URL
  SUPABASE_KEY           — Your Supabase anon/service key

All functions degrade gracefully (no-op) if disabled or package missing.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_ENABLED = os.getenv("ENABLE_SUPABASE", "false").lower() in ("true", "1", "yes")
_SUPABASE_URL = os.getenv("SUPABASE_URL", "")
_SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
_client = None


def _get_client():
    global _client
    if _client is not None:
        return _client
    if not _ENABLED:
        return None
    if not _SUPABASE_URL or not _SUPABASE_KEY:
        logger.warning("[Supabase] ENABLE_SUPABASE=true but SUPABASE_URL/KEY not set.")
        return None
    try:
        from supabase import create_client  # type: ignore
        _client = create_client(_SUPABASE_URL, _SUPABASE_KEY)
        logger.info("[Supabase] Client initialized.")
        return _client
    except ImportError:
        logger.warning("[Supabase] 'supabase' package not installed. Disabled.")
        return None
    except Exception as exc:
        logger.warning("[Supabase] Connection failed: %s", exc)
        return None


def sb_log_episode(episode_id: int, task: str, seed: int, final_score: float,
                   cumulative_reward: float, total_steps: int,
                   agent_mode: str = "unknown", extra: Optional[Dict[str, Any]] = None) -> None:
    client = _get_client()
    if client is None:
        return
    try:
        client.table("episodes").upsert({
            "local_episode_id": episode_id, "task": task, "seed": seed,
            "final_score": final_score, "cumulative_reward": cumulative_reward,
            "total_steps": total_steps, "agent_mode": agent_mode, **(extra or {}),
        }).execute()
    except Exception as exc:
        logger.warning("[Supabase] Failed to log episode: %s", exc)


def sb_log_action(episode_id: int, step: int, action_type: str,
                  ticket_id: int, reward: float, reason: str) -> None:
    client = _get_client()
    if client is None:
        return
    try:
        client.table("actions").insert({
            "local_episode_id": episode_id, "step": step,
            "action_type": action_type, "ticket_id": ticket_id,
            "reward": reward, "reason": reason,
        }).execute()
    except Exception as exc:
        logger.warning("[Supabase] Failed to log action: %s", exc)


def sb_log_metrics(episode_id: int, metrics: Dict[str, Any]) -> None:
    client = _get_client()
    if client is None:
        return
    try:
        client.table("metrics").insert(
            {"local_episode_id": episode_id, **metrics}
        ).execute()
    except Exception as exc:
        logger.warning("[Supabase] Failed to log metrics: %s", exc)


def is_enabled() -> bool:
    return _get_client() is not None
