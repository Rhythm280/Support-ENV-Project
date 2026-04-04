"""
models.py — Typed data contracts for SupportEnv

All environment I/O is fully typed with Pydantic v2.
Agents, graders, and the API all share these models.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────

class ActionType(str, Enum):
    CLASSIFY   = "classify"
    PRIORITIZE = "prioritize"
    RESPOND    = "respond"
    ESCALATE   = "escalate"
    CLOSE      = "close"
    SEARCH     = "search"


class Category(str, Enum):
    BILLING   = "billing"
    TECHNICAL = "technical"
    GENERAL   = "general"
    COMPLAINT = "complaint"
    POSITIVE  = "positive"


class Priority(str, Enum):
    LOW    = "low"
    MEDIUM = "medium"
    HIGH   = "high"


class TicketStatus(str, Enum):
    OPEN      = "open"
    CLOSED    = "closed"
    ESCALATED = "escalated"


class TaskName(str, Enum):
    EASY   = "easy"
    MEDIUM = "medium"
    HARD   = "hard"


# ─────────────────────────────────────────────
# Core domain models
# ─────────────────────────────────────────────

class Ticket(BaseModel):
    """A single customer support ticket with full metadata."""
    id: int
    text: str

    # Ground-truth labels (hidden from agent, used by grader)
    true_category: Category
    true_priority: Priority
    requires_escalation: bool = False

    # Rich grading metadata (from generator.py)
    required_keywords: List[str] = Field(default_factory=list)
    forbidden_phrases: List[str] = Field(default_factory=list)
    resolution_steps: List[str] = Field(default_factory=list)

    # User persona: angry | polite | confused
    persona: str = "polite"

    # Dynamics: ticket dependencies and auto-escalation threshold
    dependencies: List[int] = Field(
        default_factory=list,
        description="IDs of tickets that must be resolved before this one (Hard task)",
    )
    escalation_step: int = Field(
        default=10,
        description="Step number at which this ticket's priority auto-escalates",
    )

    # Agent-assigned fields (start as None, assigned during episode)
    category: Optional[Category] = None
    priority: Optional[Priority] = None
    response: Optional[str] = None
    status: TicketStatus = TicketStatus.OPEN

    # Internal: track bad-response count for tone worsening
    bad_response_count: int = Field(default=0, exclude=True)

    def to_agent_view(self) -> Dict[str, Any]:
        """Return only fields the agent is allowed to see (no ground truth)."""
        return {
            "id": self.id,
            "text": self.text,
            "persona": self.persona,
            "category": self.category.value if self.category else None,
            "priority": self.priority.value if self.priority else None,
            "status": self.status.value,
            "urgency_hint": "urgent" if self.requires_escalation else "normal",
            "dependencies": self.dependencies,
        }


# ─────────────────────────────────────────────
# Action / Observation / Result
# ─────────────────────────────────────────────

class Action(BaseModel):
    """An action submitted by an agent for one step."""
    action_type: ActionType
    ticket_id: int
    content: Optional[str] = Field(
        default=None,
        description="Category name, priority name, response text, or None for close/escalate",
    )


class Observation(BaseModel):
    """What the agent sees after each step."""
    tickets: List[Dict[str, Any]]   # agent-view dicts (no ground truth)
    step_count: int
    cumulative_reward: float
    last_action_error: bool = False
    task: str
    search_results: List[Dict[str, str]] = Field(default_factory=list)
    info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Dense metrics: tickets_resolved, efficiency, loops_detected, priority_misses",
    )


class StepResult(BaseModel):
    """Full result of an env step (for API consistency)."""
    observation: Observation
    reward: float
    done: bool
    truncated: bool = False
    info: Dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────────
# Task config
# ─────────────────────────────────────────────

class TaskConfig(BaseModel):
    """Configuration for a specific task difficulty."""
    name: TaskName
    max_steps: int
    allowed_actions: List[ActionType]
    ticket_count: int
    description: str
    success_criteria: str = ""
