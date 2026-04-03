"""
database.py — SQLite persistence layer for SupportEnv.

Uses stdlib sqlite3 (no extra dependencies).

Tables:
  episodes  — one row per episode (task, seed, timestamps, final score)
  tickets   — ticket snapshot at end of episode
  actions   — every step taken during an episode
  metrics   — summary metrics per episode

Supports:
  - init_db()         — create tables if not exist
  - log_episode()     — insert/update episode record
  - log_action()      — append action to episode log
  - log_metric()      — save final metrics
  - load_episode()    — fetch full episode for replay
  - list_episodes()   — list all episodes with basic stats
  - replay_episode()  — reconstruct action sequence for replay
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Generator, List, Optional

logger = logging.getLogger(__name__)

# Default DB path — can override via env var
DB_PATH = os.getenv("SUPPORT_ENV_DB", "support_env.db")


# ─────────────────────────────────────────────────────────────────────────────
# Connection management
# ─────────────────────────────────────────────────────────────────────────────

@contextmanager
def _get_conn(db_path: str = DB_PATH) -> Generator[sqlite3.Connection, None, None]:
    """Context manager for SQLite connections with WAL mode enabled."""
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# Schema
# ─────────────────────────────────────────────────────────────────────────────

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS episodes (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    task           TEXT    NOT NULL,
    seed           INTEGER NOT NULL,
    started_at     TEXT    NOT NULL,
    ended_at       TEXT,
    total_steps    INTEGER DEFAULT 0,
    final_score    REAL    DEFAULT 0.0,
    cumulative_reward REAL DEFAULT 0.0,
    done           INTEGER DEFAULT 0,
    agent_mode     TEXT    DEFAULT 'unknown',
    extra_json     TEXT    DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS tickets (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id     INTEGER NOT NULL REFERENCES episodes(id),
    ticket_id      INTEGER NOT NULL,
    text           TEXT    NOT NULL,
    true_category  TEXT    NOT NULL,
    true_priority  TEXT    NOT NULL,
    assigned_category TEXT,
    assigned_priority TEXT,
    response       TEXT,
    status         TEXT    DEFAULT 'open',
    requires_escalation INTEGER DEFAULT 0,
    persona        TEXT,
    required_keywords TEXT,
    forbidden_phrases TEXT,
    resolution_steps  TEXT
);

CREATE TABLE IF NOT EXISTS actions (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id     INTEGER NOT NULL REFERENCES episodes(id),
    step           INTEGER NOT NULL,
    action_type    TEXT    NOT NULL,
    ticket_id      INTEGER NOT NULL,
    content        TEXT,
    reward         REAL    NOT NULL,
    cumulative_reward REAL,
    reason         TEXT,
    timestamp      TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS metrics (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id       INTEGER NOT NULL REFERENCES episodes(id),
    tickets_resolved INTEGER DEFAULT 0,
    efficiency       REAL    DEFAULT 0.0,
    loops_detected   INTEGER DEFAULT 0,
    priority_misses  INTEGER DEFAULT 0,
    final_score      REAL    DEFAULT 0.0,
    breakdown_json   TEXT    DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_actions_episode ON actions(episode_id);
CREATE INDEX IF NOT EXISTS idx_tickets_episode ON tickets(episode_id);
CREATE INDEX IF NOT EXISTS idx_metrics_episode ON metrics(episode_id);
"""


def init_db(db_path: str = DB_PATH) -> None:
    """Create all tables if they don't exist. Safe to call multiple times."""
    with _get_conn(db_path) as conn:
        conn.executescript(_SCHEMA_SQL)
    logger.info("[DB] Initialized SQLite database at: %s", db_path)


# ─────────────────────────────────────────────────────────────────────────────
# Episode logging
# ─────────────────────────────────────────────────────────────────────────────

def log_episode_start(task: str, seed: int, agent_mode: str = "unknown", db_path: str = DB_PATH) -> int:
    """
    Insert a new episode row and return its ID.
    Call this at the beginning of reset().
    """
    now = datetime.now(timezone.utc).isoformat()
    with _get_conn(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO episodes (task, seed, started_at, agent_mode) VALUES (?, ?, ?, ?)",
            (task, seed, now, agent_mode),
        )
        episode_id = cur.lastrowid
    logger.debug("[DB] Episode started: id=%d task=%s seed=%d", episode_id, task, seed)
    return episode_id


def log_episode_end(
    episode_id: int,
    total_steps: int,
    final_score: float,
    cumulative_reward: float,
    done: bool,
    db_path: str = DB_PATH,
) -> None:
    """Update episode row with final stats. Call this after episode completes."""
    now = datetime.now(timezone.utc).isoformat()
    with _get_conn(db_path) as conn:
        conn.execute(
            """UPDATE episodes
               SET ended_at=?, total_steps=?, final_score=?, cumulative_reward=?, done=?
               WHERE id=?""",
            (now, total_steps, final_score, cumulative_reward, int(done), episode_id),
        )
    logger.debug("[DB] Episode ended: id=%d score=%.3f", episode_id, final_score)


def log_tickets(episode_id: int, tickets: List[Any], db_path: str = DB_PATH) -> None:
    """Snapshot all ticket states at the end of an episode."""
    rows = []
    for t in tickets:
        rows.append((
            episode_id,
            t.id,
            t.text,
            t.true_category.value if hasattr(t.true_category, "value") else str(t.true_category),
            t.true_priority.value if hasattr(t.true_priority, "value") else str(t.true_priority),
            t.category.value if t.category else None,
            t.priority.value if t.priority else None,
            t.response,
            t.status.value if hasattr(t.status, "value") else str(t.status),
            int(t.requires_escalation),
            getattr(t, "persona", "unknown"),
            json.dumps(getattr(t, "required_keywords", [])),
            json.dumps(getattr(t, "forbidden_phrases", [])),
            json.dumps(getattr(t, "resolution_steps", [])),
        ))
    with _get_conn(db_path) as conn:
        conn.executemany(
            """INSERT INTO tickets
               (episode_id, ticket_id, text, true_category, true_priority,
                assigned_category, assigned_priority, response, status,
                requires_escalation, persona, required_keywords, forbidden_phrases, resolution_steps)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            rows,
        )


def log_action(
    episode_id: int,
    step: int,
    action_type: str,
    ticket_id: int,
    content: Optional[str],
    reward: float,
    cumulative_reward: float,
    reason: str,
    db_path: str = DB_PATH,
) -> None:
    """Log a single action step. Called by env.step() on every transition."""
    now = datetime.now(timezone.utc).isoformat()
    with _get_conn(db_path) as conn:
        conn.execute(
            """INSERT INTO actions
               (episode_id, step, action_type, ticket_id, content, reward,
                cumulative_reward, reason, timestamp)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (episode_id, step, action_type, ticket_id, content,
             round(reward, 4), round(cumulative_reward, 4), reason, now),
        )


def log_metrics(
    episode_id: int,
    tickets_resolved: int,
    efficiency: float,
    loops_detected: int,
    priority_misses: int,
    final_score: float,
    breakdown: Dict[str, float],
    db_path: str = DB_PATH,
) -> None:
    """Save computed metrics after grading. Called once per episode."""
    with _get_conn(db_path) as conn:
        conn.execute(
            """INSERT INTO metrics
               (episode_id, tickets_resolved, efficiency, loops_detected,
                priority_misses, final_score, breakdown_json)
               VALUES (?,?,?,?,?,?,?)""",
            (
                episode_id,
                tickets_resolved,
                round(efficiency, 4),
                loops_detected,
                priority_misses,
                round(final_score, 4),
                json.dumps(breakdown),
            ),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Query / Replay
# ─────────────────────────────────────────────────────────────────────────────

def list_episodes(limit: int = 20, db_path: str = DB_PATH) -> List[Dict[str, Any]]:
    """List recent episodes with basic stats."""
    with _get_conn(db_path) as conn:
        rows = conn.execute(
            """SELECT e.id, e.task, e.seed, e.agent_mode, e.started_at, e.ended_at,
                      e.total_steps, e.final_score, e.cumulative_reward, e.done
               FROM episodes e
               ORDER BY e.id DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def load_episode(episode_id: int, db_path: str = DB_PATH) -> Dict[str, Any]:
    """
    Load a complete episode (episode + tickets + actions + metrics).
    Used for replay and analysis.
    """
    with _get_conn(db_path) as conn:
        ep_row = conn.execute(
            "SELECT * FROM episodes WHERE id=?", (episode_id,)
        ).fetchone()
        if ep_row is None:
            raise ValueError(f"Episode {episode_id} not found in database.")

        action_rows = conn.execute(
            "SELECT * FROM actions WHERE episode_id=? ORDER BY step ASC",
            (episode_id,),
        ).fetchall()

        ticket_rows = conn.execute(
            "SELECT * FROM tickets WHERE episode_id=? ORDER BY ticket_id ASC",
            (episode_id,),
        ).fetchall()

        metric_rows = conn.execute(
            "SELECT * FROM metrics WHERE episode_id=?", (episode_id,)
        ).fetchall()

    return {
        "episode": dict(ep_row),
        "tickets": [dict(r) for r in ticket_rows],
        "actions": [dict(r) for r in action_rows],
        "metrics": [dict(r) for r in metric_rows],
    }


def replay_episode(episode_id: int, db_path: str = DB_PATH) -> List[Dict[str, Any]]:
    """
    Reconstruct the action sequence for a given episode.

    Returns a list of action dicts in chronological order.
    These can be re-fed into a new env instance to reproduce the episode.
    """
    data = load_episode(episode_id, db_path)
    episode = data["episode"]
    actions = data["actions"]

    logger.info(
        "[REPLAY] Episode %d | Task: %s | Seed: %d | Steps: %d | Score: %.3f",
        episode_id,
        episode["task"],
        episode["seed"],
        episode["total_steps"],
        episode["final_score"],
    )

    for action in actions:
        logger.debug(
            "[REPLAY] Step %d | %s on ticket #%d | reward=%.3f | %s",
            action["step"],
            action["action_type"],
            action["ticket_id"],
            action["reward"],
            action["reason"],
        )

    return actions
