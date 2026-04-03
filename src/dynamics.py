"""
dynamics.py — Dynamic state evolution rules for SupportEnv.

Implements:
  - Priority escalation over time (tickets become more urgent after N steps)
  - User tone worsening on bad/no responses
  - Ticket dependencies (some tickets block others)
  - Partial observability helper (strips ground truth from agent view)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.models import Priority, Ticket, TicketStatus

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Priority Escalation
# ─────────────────────────────────────────────────────────────────────────────

_PRIORITY_ESCALATION_ORDER = [Priority.LOW, Priority.MEDIUM, Priority.HIGH]


def apply_priority_escalation(tickets: List[Ticket], current_step: int) -> List[Ticket]:
    """
    Escalate ticket priority if the episode has exceeded the ticket's escalation_step.

    Only escalates tickets that are still open and haven't been manually assigned
    a priority higher than the escalation target.

    Returns the list of tickets (mutated in-place) with any escalated ones noted.
    """
    escalated = []
    for ticket in tickets:
        if ticket.status != TicketStatus.OPEN:
            continue
        esc_threshold = getattr(ticket, "escalation_step", 10)
        if current_step < esc_threshold:
            continue

        # Find current priority index
        current_idx = _PRIORITY_ESCALATION_ORDER.index(ticket.true_priority)
        if current_idx < len(_PRIORITY_ESCALATION_ORDER) - 1:
            new_priority = _PRIORITY_ESCALATION_ORDER[current_idx + 1]
            if new_priority != ticket.true_priority:
                logger.debug(
                    "[Dynamics] Ticket #%d priority escalated: %s → %s (step=%d, threshold=%d)",
                    ticket.id, ticket.true_priority, new_priority, current_step, esc_threshold,
                )
                ticket.true_priority = new_priority  # mutate ground truth
                escalated.append(ticket.id)

    return tickets


# ─────────────────────────────────────────────────────────────────────────────
# User Tone Worsening
# ─────────────────────────────────────────────────────────────────────────────

_TONE_PROGRESSION = ["polite", "confused", "angry"]


def worsen_tone(ticket: Ticket, bad_response_count: int) -> str:
    """
    Worsen user tone based on number of bad/irrelevant responses.

    Returns the new persona string. Side-effects: appends angry suffix to ticket.text.
    """
    current_persona = getattr(ticket, "persona", "polite")
    if bad_response_count == 0:
        return current_persona

    try:
        current_idx = _TONE_PROGRESSION.index(current_persona)
    except ValueError:
        current_idx = 0

    new_idx = min(current_idx + bad_response_count, len(_TONE_PROGRESSION) - 1)
    new_persona = _TONE_PROGRESSION[new_idx]

    if new_persona != current_persona:
        ticket.persona = new_persona
        angry_suffix = " This is outrageous! I demand an immediate response!!"
        if not ticket.text.endswith(angry_suffix):
            ticket.text += angry_suffix
        logger.debug(
            "[Dynamics] Ticket #%d tone worsened: %s → %s (bad_responses=%d)",
            ticket.id, current_persona, new_persona, bad_response_count,
        )

    return new_persona


# ─────────────────────────────────────────────────────────────────────────────
# Ticket Dependencies
# ─────────────────────────────────────────────────────────────────────────────

def assign_dependencies(tickets: List[Ticket], seed: int) -> List[Ticket]:
    """
    Assign dependencies between tickets for the Hard task.

    Rules:
      - At most 2 tickets can have dependencies.
      - A ticket's dependency must always have a lower ID (no circular deps).
      - Dependencies are deterministic per seed.

    Mutates tickets in-place and returns them.
    """
    import random
    rng = random.Random(seed + 999)  # offset seed to get different pattern than ticket ordering

    if len(tickets) < 3:
        return tickets  # not enough tickets to create meaningful deps

    # Choose 1 dependency pair
    dependent_idx = rng.randint(2, len(tickets) - 1)   # at least the 3rd ticket
    blocker_idx = rng.randint(0, dependent_idx - 1)    # must be earlier

    tickets[dependent_idx].dependencies = [tickets[blocker_idx].id]
    logger.debug(
        "[Dynamics] Dependency assigned: ticket #%d depends on ticket #%d",
        tickets[dependent_idx].id, tickets[blocker_idx].id,
    )

    return tickets


def check_dependencies_met(ticket: Ticket, all_tickets: List[Ticket]) -> bool:
    """
    Return True if all tickets that `ticket` depends on are already resolved.
    Agents should not act on a ticket until its dependencies are met.
    """
    if not getattr(ticket, "dependencies", []):
        return True

    dep_ids = ticket.dependencies
    for dep_id in dep_ids:
        dep = next((t for t in all_tickets if t.id == dep_id), None)
        if dep is None:
            continue
        if dep.status not in (TicketStatus.CLOSED, TicketStatus.ESCALATED):
            return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Partial Observability
# ─────────────────────────────────────────────────────────────────────────────

def partial_observation(ticket: Ticket, reveal_deps: bool = True) -> Dict[str, Any]:
    """
    Return the agent-facing view of a ticket.

    Hides:
      - true_category
      - true_priority
      - requires_escalation (partially hidden — only shown as urgency hint)
      - resolution_steps

    Reveals:
      - id, text, persona, assigned category/priority, status
      - dependencies (so agent knows what to resolve first)
      - urgency_hint (derived from requires_escalation, not literal)
    """
    urgency_hint = "urgent" if ticket.requires_escalation else "normal"

    view: Dict[str, Any] = {
        "id": ticket.id,
        "text": ticket.text,
        "persona": getattr(ticket, "persona", "unknown"),
        "category": ticket.category.value if ticket.category else None,
        "priority": ticket.priority.value if ticket.priority else None,
        "status": ticket.status.value,
        "urgency_hint": urgency_hint,
    }

    if reveal_deps:
        view["dependencies"] = getattr(ticket, "dependencies", [])
        if ticket.dependencies:
            view["dependency_note"] = f"Must resolve ticket(s) {ticket.dependencies} first."

    return view
