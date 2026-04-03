"""
tasks.py — Task configurations and deterministic graders.

Three tasks of increasing difficulty:
  easy   → classify tickets only
  medium → classify + prioritize
  hard   → full pipeline (classify → prioritize → respond/escalate → close)

Grading delegates to src/graders.py for keyword-based scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from src.models import ActionType, TaskConfig, TaskName


# ─────────────────────────────────────────────
# Task definitions
# ─────────────────────────────────────────────

TASK_CONFIGS: dict[TaskName, TaskConfig] = {
    TaskName.EASY: TaskConfig(
        name=TaskName.EASY,
        max_steps=15,
        allowed_actions=[ActionType.CLASSIFY],
        ticket_count=5,
        description=(
            "Classify 5 customer tickets into the correct category. "
            "Each correct classification earns +0.30. Wrong costs -0.30. "
            "Repeated actions incur -0.50 loop penalty."
        ),
        success_criteria=(
            "Score ≥ 0.8 (at least 4/5 tickets correctly classified). "
            "No repeated actions on the same ticket."
        ),
    ),
    TaskName.MEDIUM: TaskConfig(
        name=TaskName.MEDIUM,
        max_steps=25,
        allowed_actions=[ActionType.CLASSIFY, ActionType.PRIORITIZE],
        ticket_count=6,
        description=(
            "Classify and prioritize 6 tickets. "
            "Each action is graded separately — you need both to score full points. "
            "Score = 0.5 × classification_rate + 0.5 × priority_rate."
        ),
        success_criteria=(
            "Score ≥ 0.75 (correct classification AND correct priority on most tickets). "
            "Avoid repeated classify/prioritize on the same ticket."
        ),
    ),
    TaskName.HARD: TaskConfig(
        name=TaskName.HARD,
        max_steps=40,
        allowed_actions=[
            ActionType.CLASSIFY,
            ActionType.PRIORITIZE,
            ActionType.RESPOND,
            ActionType.ESCALATE,
            ActionType.CLOSE,
        ],
        ticket_count=7,
        description=(
            "Full resolution pipeline: classify → prioritize → respond or escalate → close. "
            "Response quality is graded by required_keywords and forbidden_phrase checks. "
            "Ticket dependencies must be respected. Priority auto-escalates over time."
        ),
        success_criteria=(
            "Score ≥ 0.70 across all four axes: classification (30%), prioritization (25%), "
            "response quality (25%), escalation accuracy (20%). "
            "Handle dependencies correctly. Avoid loops. Close all tickets efficiently."
        ),
    ),
}


# ─────────────────────────────────────────────
# GradeReport (compat shim — delegates to graders.py)
# ─────────────────────────────────────────────

@dataclass
class GradeReport:
    """Simplified grade report for backwards compatibility."""
    score: float                  # 0.0 – 1.0
    breakdown: dict               # per-axis sub-scores
    summary: str


def grade(task: TaskName, tickets) -> GradeReport:
    """Route to the correct grader via src/graders.py."""
    from src.graders import grade as _grade
    report = _grade(task.value, tickets)
    return GradeReport(
        score=report.score,
        breakdown=report.breakdown,
        summary=report.summary,
    )
