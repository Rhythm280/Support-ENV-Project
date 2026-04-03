"""
rewards.py — All reward logic in one place.

Dense reward shaping per spec:
  +0.3  correct classification
  +0.5  correct/relevant resolution response
  +0.2  no forbidden phrases in response (new)
  -0.3  wrong action (wrong category/priority/response)
  -0.5  repeated action on same ticket (loop detection)
  -1.0  invalid action (disallowed type or unknown ticket)
"""

from __future__ import annotations

from src.models import Action, ActionType, Category, Priority, Ticket


# ─────────────────────────────────────────────
# Reward constants — per spec
# ─────────────────────────────────────────────

R_CLASSIFY_CORRECT   = +0.30
R_CLASSIFY_WRONG     = -0.30
R_PRIORITIZE_CORRECT = +0.30
R_PRIORITIZE_WRONG   = -0.30
R_RESPOND_RELEVANT   = +0.50
R_RESPOND_IRRELEVANT = -0.30
R_NO_FORBIDDEN_BONUS = +0.20   # bonus for clean response (no forbidden phrases)
R_ESCALATE_CORRECT   = +0.30
R_ESCALATE_WRONG     = -0.30
R_CLOSE_BONUS        = +0.20   # bonus for fully resolving before closing
R_WRONG_ACTION       = -0.30   # generic wrong action
R_REPEATED_ACTION    = -0.50   # loop penalty: same action on same ticket twice
R_INVALID_ACTION     = -1.00   # unknown ticket ID or completely disallowed action


# ─────────────────────────────────────────────
# Response relevance heuristics
# ─────────────────────────────────────────────

_CATEGORY_KEYWORDS: dict[Category, list[str]] = {
    Category.BILLING:   ["refund", "charge", "invoice", "payment", "billing",
                         "subscription", "cancel", "price", "cost", "fee"],
    Category.TECHNICAL: ["crash", "error", "bug", "slow", "login", "api", "broken",
                         "fix", "issue", "reset", "password", "blank", "404", "500"],
    Category.GENERAL:   ["how", "what", "when", "where", "why", "can i", "team",
                         "hours", "discount", "feature", "update", "add"],
    Category.COMPLAINT: ["sorry", "apologies", "understand", "frustration", "escalate",
                         "manager", "resolve", "compensate", "priority"],
    Category.POSITIVE:  ["thank", "glad", "happy", "appreciate", "welcome", "feedback", "noted"],
}


def _response_is_relevant(ticket: Ticket, response: str) -> bool:
    """Check if response text contains keywords relevant to the ticket's true category."""
    if not response:
        return False
    lower = response.lower()
    keywords = _CATEGORY_KEYWORDS.get(ticket.true_category, [])
    # Also check ticket-specific required_keywords if present
    required = getattr(ticket, "required_keywords", [])
    combined = list(set(keywords + required))
    hits = sum(1 for kw in combined if kw in lower)
    return hits >= 1


def _has_forbidden_phrases(ticket: Ticket, response: str) -> bool:
    """Return True if response contains any forbidden phrases."""
    if not response:
        return False
    lower = response.lower()
    forbidden = getattr(ticket, "forbidden_phrases", [])
    return any(phrase in lower for phrase in forbidden)


# ─────────────────────────────────────────────
# Public reward calculator
# ─────────────────────────────────────────────

def calculate_reward(
    ticket: Ticket,
    action: Action,
    action_history: list[tuple[str, int]] | None = None,
) -> tuple[float, str]:
    """
    Calculate reward for an (action, ticket) pair.

    Args:
        ticket:         The target ticket.
        action:         The action taken.
        action_history: List of (action_type, ticket_id) tuples from previous steps.
                        Used for loop/repeat detection.

    Returns:
        (reward: float, reason: str)
    """
    atype = action.action_type
    history = action_history or []

    # ── Repeated action detection (loop penalty) ──────────────────────────────
    prev_same = sum(
        1 for (at, tid) in history
        if at == atype.value and tid == action.ticket_id
    )
    if prev_same >= 1:
        return R_REPEATED_ACTION, f"✗ Repeated {atype.value} on ticket #{action.ticket_id} (loop)"

    # ── CLASSIFY ──────────────────────────────────────────────────────────────
    if atype == ActionType.CLASSIFY:
        try:
            guess = Category(action.content)
        except (ValueError, TypeError):
            return R_CLASSIFY_WRONG, f"Invalid category '{action.content}'"
        if guess == ticket.true_category:
            return R_CLASSIFY_CORRECT, f"✓ Correct category: {guess.value}"
        return R_CLASSIFY_WRONG, f"✗ Expected {ticket.true_category.value}, got {guess.value}"

    # ── PRIORITIZE ────────────────────────────────────────────────────────────
    elif atype == ActionType.PRIORITIZE:
        try:
            guess = Priority(action.content)
        except (ValueError, TypeError):
            return R_PRIORITIZE_WRONG, f"Invalid priority '{action.content}'"
        if guess == ticket.true_priority:
            return R_PRIORITIZE_CORRECT, f"✓ Correct priority: {guess.value}"
        return R_PRIORITIZE_WRONG, f"✗ Expected {ticket.true_priority.value}, got {guess.value}"

    # ── RESPOND ───────────────────────────────────────────────────────────────
    elif atype == ActionType.RESPOND:
        response_text = action.content or ""
        relevant = _response_is_relevant(ticket, response_text)
        forbidden = _has_forbidden_phrases(ticket, response_text)

        if relevant and not forbidden:
            # Both signals pass: full reward + no-forbidden bonus
            return R_RESPOND_RELEVANT + R_NO_FORBIDDEN_BONUS, "✓ Relevant response, no forbidden phrases"
        elif relevant and forbidden:
            # Relevant but contains forbidden phrase: no bonus
            return R_RESPOND_RELEVANT, "⚠ Relevant response, but contains forbidden phrase"
        elif not relevant and not forbidden:
            # Not relevant but at least safe
            return R_RESPOND_IRRELEVANT + R_NO_FORBIDDEN_BONUS, "✗ Response missing relevant keywords (but safe)"
        else:
            return R_RESPOND_IRRELEVANT, "✗ Irrelevant response with forbidden phrase"

    # ── ESCALATE ──────────────────────────────────────────────────────────────
    elif atype == ActionType.ESCALATE:
        if ticket.requires_escalation:
            return R_ESCALATE_CORRECT, "✓ Correct escalation (ticket needed it)"
        return R_ESCALATE_WRONG, "✗ False escalation (ticket didn't need it)"

    # ── CLOSE ─────────────────────────────────────────────────────────────────
    elif atype == ActionType.CLOSE:
        fully_handled = (
            ticket.category is not None
            and ticket.priority is not None
            and (ticket.response is not None or ticket.status.value == "escalated")
        )
        if fully_handled:
            return R_CLOSE_BONUS, "✓ Ticket fully resolved before closing"
        return 0.0, "ℹ Closed without full resolution (no penalty)"

    return R_INVALID_ACTION, f"✗ Unknown action type: {atype}"
