"""
graders.py — Deterministic, keyword-based graders for SupportEnv.

Scoring breakdown per ticket (returns float strictly in (0, 1)):
  +0.3  category match (correct classification)
  +0.5  required_keywords present in response
  +0.2  no forbidden_phrases in response

These graders are PURELY rule-based — NO LLM calls, NO stochasticity.
All scores are strictly in (0.01, 0.99) — never exactly 0.0 or 1.0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.models import Category, Ticket


# ─────────────────────────────────────────────────────────────────────────────
# Score weights (must sum to 1.0)
# ─────────────────────────────────────────────────────────────────────────────

W_CATEGORY_MATCH    = 0.3
W_REQUIRED_KEYWORDS = 0.5
W_NO_FORBIDDEN      = 0.2

# Validator requires scores strictly inside (0, 1) — not 0.0, not 1.0
_SCORE_MIN = 0.01
_SCORE_MAX = 0.99

def _clamp(score: float) -> float:
    """
    Clamp score strictly into (_SCORE_MIN, _SCORE_MAX) = (0.01, 0.99).
    NEVER returns exactly 0.0 or 1.0.

    Handles: NaN, infinity, None, integer 0/1, and all float edge cases.
    """
    try:
        s = float(score)
    except (TypeError, ValueError):
        return _SCORE_MIN

    # Guard against NaN (NaN != NaN is the only reliable NaN check)
    if s != s:
        return _SCORE_MIN

    # Explicit boundary check — never 0.0 or 1.0
    if s <= 0.0:
        return _SCORE_MIN
    if s >= 1.0:
        return _SCORE_MAX

    # Round to avoid floating-point creep (e.g. 0.99999... rounding up)
    s = round(s, 4)

    # Final hard clamp after rounding
    if s <= 0.0:
        return _SCORE_MIN
    if s >= 1.0:
        return _SCORE_MAX

    return s


@dataclass
class TicketGradeResult:
    """Per-ticket grade breakdown."""
    ticket_id: int
    score: float                    # 0.0 – 1.0
    category_match: bool
    keywords_present: bool
    no_forbidden: bool
    missing_keywords: List[str] = field(default_factory=list)
    found_forbidden: List[str] = field(default_factory=list)
    detail: str = ""


@dataclass
class EpisodeGradeReport:
    """Aggregate grading report for a full episode."""
    score: float                     # 0.0 – 1.0 (weighted average)
    breakdown: Dict[str, float]      # per-axis sub-scores
    ticket_results: List[TicketGradeResult]
    summary: str


# ─────────────────────────────────────────────────────────────────────────────
# Per-ticket grader
# ─────────────────────────────────────────────────────────────────────────────

def grade_ticket(ticket: Ticket, response_text: Optional[str] = None) -> TicketGradeResult:
    """
    Grade a single ticket based on three deterministic signals:

    1. Category match          → +0.3 (was ticket correctly classified?)
    2. Required keywords       → +0.5 (does the response contain them?)
    3. No forbidden phrases    → +0.2 (does the response avoid bad phrases?)

    Args:
        ticket:        The Ticket object with true labels and metadata.
        response_text: The agent's response text (or None if not yet responded).

    Returns:
        TicketGradeResult with score and breakdown.
    """
    score = 0.0
    response_lower = (response_text or "").lower()

    # ── Signal 1: Category match ──────────────────────────────────────────────
    category_match = (ticket.category is not None and ticket.category == ticket.true_category)
    if category_match:
        score += W_CATEGORY_MATCH

    # ── Signal 2: Required keywords in response ────────────────────────────────
    missing_keywords: List[str] = []
    if response_text and ticket.required_keywords:
        missing_keywords = [kw for kw in ticket.required_keywords if kw not in response_lower]
        keywords_present = len(missing_keywords) == 0
    else:
        keywords_present = False
        missing_keywords = ticket.required_keywords[:]

    if keywords_present:
        score += W_REQUIRED_KEYWORDS

    # ── Signal 3: No forbidden phrases ────────────────────────────────────────
    found_forbidden: List[str] = []
    if ticket.forbidden_phrases:
        found_forbidden = [phrase for phrase in ticket.forbidden_phrases if phrase in response_lower]
        no_forbidden = len(found_forbidden) == 0
    else:
        no_forbidden = True

    if no_forbidden:
        score += W_NO_FORBIDDEN

    # ── Build detail string ───────────────────────────────────────────────────
    parts = []
    parts.append(f"category={'✓' if category_match else '✗'}")
    parts.append(f"keywords={'✓' if keywords_present else f'✗ missing={missing_keywords}'}")
    parts.append(f"safety={'✓' if no_forbidden else f'✗ found={found_forbidden}'}")

    return TicketGradeResult(
        ticket_id=ticket.id,
        score=_clamp(score),
        category_match=category_match,
        keywords_present=keywords_present,
        no_forbidden=no_forbidden,
        missing_keywords=missing_keywords,
        found_forbidden=found_forbidden,
        detail=" | ".join(parts),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Task-level graders
# ─────────────────────────────────────────────────────────────────────────────

def grade_easy(tickets: List[Ticket]) -> EpisodeGradeReport:
    """
    Easy Task: Score = fraction of tickets correctly classified.
    
    Only Signal 1 (category match) counts.
    Score = correct_classifications / total_tickets
    """
    total = len(tickets)
    if not total:
        return EpisodeGradeReport(
            score=_SCORE_MIN,
            breakdown={"classification": _SCORE_MIN},
            ticket_results=[],
            summary="No tickets"
        )

    ticket_results = []
    correct = 0
    for t in tickets:
        match = (t.category is not None and t.category == t.true_category)
        if match:
            correct += 1
        ticket_results.append(TicketGradeResult(
            ticket_id=t.id,
            score=_clamp(1.0) if match else _SCORE_MIN,
            category_match=match,
            keywords_present=False,
            no_forbidden=True,
            detail=f"classify={'✓' if match else '✗'} (got={t.category}, expected={t.true_category})",
        ))

    score = _clamp(correct / total)
    return EpisodeGradeReport(
        score=score,
        breakdown={"classification": score},
        ticket_results=ticket_results,
        summary=f"{correct}/{total} tickets correctly classified → {score:.0%}",
    )


def grade_medium(tickets: List[Ticket]) -> EpisodeGradeReport:
    """
    Medium Task: Score = 0.5 × classification_rate + 0.5 × priority_rate.
    Uses both category match and priority match.
    """
    total = len(tickets)
    if not total:
        return EpisodeGradeReport(
            score=_SCORE_MIN,
            breakdown={"classification": _SCORE_MIN, "prioritization": _SCORE_MIN},
            ticket_results=[],
            summary="No tickets"
        )

    ticket_results = []
    correct_cats = 0
    correct_pris = 0

    for t in tickets:
        cat_match = (t.category is not None and t.category == t.true_category)
        pri_match = (t.priority is not None and t.priority == t.true_priority)
        if cat_match:
            correct_cats += 1
        if pri_match:
            correct_pris += 1

        partial_score = (0.5 if cat_match else 0.0) + (0.5 if pri_match else 0.0)
        ticket_results.append(TicketGradeResult(
            ticket_id=t.id,
            score=_clamp(round(partial_score, 3)),
            category_match=cat_match,
            keywords_present=False,
            no_forbidden=True,
            detail=f"classify={'✓' if cat_match else '✗'} | prioritize={'✓' if pri_match else '✗'}",
        ))

    cat_rate = correct_cats / total
    pri_rate = correct_pris / total
    score = _clamp(0.5 * cat_rate + 0.5 * pri_rate)

    return EpisodeGradeReport(
        score=score,
        breakdown={
            "classification": _clamp(cat_rate),
            "prioritization": _clamp(pri_rate),
        },
        ticket_results=ticket_results,
        summary=(
            f"Classification {correct_cats}/{total} ({cat_rate:.0%}), "
            f"Priority {correct_pris}/{total} ({pri_rate:.0%}) → {score:.0%}"
        ),
    )


def grade_hard(tickets: List[Ticket]) -> EpisodeGradeReport:
    """
    Hard Task: Full keyword-based grading across all signals.

    Weighted score:
      30% — classification accuracy
      25% — prioritization accuracy
      25% — response quality (required_keywords + no_forbidden_phrases)
      20% — escalation accuracy
    """
    total = len(tickets)
    if not total:
        return EpisodeGradeReport(
            score=_SCORE_MIN,
            breakdown={
                "classification": _SCORE_MIN,
                "prioritization": _SCORE_MIN,
                "response_quality": _SCORE_MIN,
                "escalation": _SCORE_MIN
            },
            ticket_results=[],
            summary="No tickets"
        )

    ticket_results = []
    correct_cats = 0
    correct_pris = 0
    response_scores: List[float] = []

    needs_esc = [t for t in tickets if t.requires_escalation]
    no_esc = [t for t in tickets if not t.requires_escalation]
    esc_correct = sum(1 for t in needs_esc if t.status.value == "escalated")
    no_esc_correct = sum(1 for t in no_esc if t.status.value != "escalated")
    esc_total = len(needs_esc) + len(no_esc)
    esc_rate = (esc_correct + no_esc_correct) / esc_total if esc_total else 0.0

    for t in tickets:
        result = grade_ticket(t, response_text=t.response)
        ticket_results.append(result)
        if result.category_match:
            correct_cats += 1
        if t.priority is not None and t.priority == t.true_priority:
            correct_pris += 1
        # Response score: keywords (5/7 weight) + no-forbidden (2/7 weight) → sums to 1.0
        # Using exact fractions avoids floating-point accumulation to exactly 1.0
        resp_score = 0.0
        if t.response:
            response_lower = t.response.lower()
            keywords_hit = all(kw in response_lower for kw in t.required_keywords)
            no_bad = not any(p in response_lower for p in t.forbidden_phrases)
            raw = (5 / 7 if keywords_hit else 0.0) + (2 / 7 if no_bad else 0.0)
            # Clamp each individual resp_score before accumulating
            resp_score = _clamp(raw)
        else:
            resp_score = _SCORE_MIN
        response_scores.append(resp_score)

    cat_rate = correct_cats / total
    pri_rate = correct_pris / total
    resp_rate = sum(response_scores) / total if response_scores else 0.0

    score = _clamp(
        0.30 * cat_rate +
        0.25 * pri_rate +
        0.25 * resp_rate +
        0.20 * esc_rate
    )

    return EpisodeGradeReport(
        score=score,
        breakdown={
            "classification": _clamp(cat_rate),
            "prioritization": _clamp(pri_rate),
            "response_quality": _clamp(resp_rate),
            "escalation": _clamp(esc_rate),
        },
        ticket_results=ticket_results,
        summary=(
            f"Cat {cat_rate:.0%} | Pri {pri_rate:.0%} | "
            f"Resp {resp_rate:.0%} | Esc {esc_rate:.0%} → {score:.0%}"
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Dispatcher
# ─────────────────────────────────────────────────────────────────────────────

def grade(task_name: str, tickets: List[Ticket]) -> EpisodeGradeReport:
    """Route to the correct grader based on task name."""
    _graders = {
        "easy":   grade_easy,
        "medium": grade_medium,
        "hard":   grade_hard,
    }
    fn = _graders.get(task_name)
    if fn is None:
        raise ValueError(f"Unknown task: '{task_name}'. Choose from: easy, medium, hard.")
    return fn(tickets)
