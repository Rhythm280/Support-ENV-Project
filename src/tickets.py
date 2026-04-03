"""
tickets.py — Rich bank of 28 realistic customer support tickets.

Each ticket ships with ground-truth category, priority, and escalation flag.
Tickets are sampled per episode so each run is different.
"""

from __future__ import annotations

from typing import List

from src.models import Category, Priority, Ticket


# ─────────────────────────────────────────────
# Ticket bank — 28 realistic scenarios
# ─────────────────────────────────────────────

_TICKET_BANK: List[dict] = [
    # ── BILLING ──────────────────────────────────────────────────────────────
    {
        "text": "I was charged twice for my subscription this month. Please refund one of the payments immediately.",
        "true_category": Category.BILLING,
        "true_priority": Priority.HIGH,
        "requires_escalation": False,
    },
    {
        "text": "My invoice shows a charge I don't recognize. Can someone explain transaction #4872?",
        "true_category": Category.BILLING,
        "true_priority": Priority.MEDIUM,
        "requires_escalation": False,
    },
    {
        "text": "I cancelled my plan 3 days ago but was still billed. I want a full refund or I'm disputing the charge.",
        "true_category": Category.BILLING,
        "true_priority": Priority.HIGH,
        "requires_escalation": True,
    },
    {
        "text": "How do I update my payment method from a debit card to a credit card?",
        "true_category": Category.BILLING,
        "true_priority": Priority.LOW,
        "requires_escalation": False,
    },
    {
        "text": "I'd like to upgrade my plan from Basic to Pro. What's the prorated cost?",
        "true_category": Category.BILLING,
        "true_priority": Priority.LOW,
        "requires_escalation": False,
    },

    # ── TECHNICAL ────────────────────────────────────────────────────────────
    {
        "text": "The app crashes every time I try to export my report to PDF. This is blocking my work.",
        "true_category": Category.TECHNICAL,
        "true_priority": Priority.HIGH,
        "requires_escalation": False,
    },
    {
        "text": "Dashboard is loading extremely slow — takes over 30 seconds on every page.",
        "true_category": Category.TECHNICAL,
        "true_priority": Priority.MEDIUM,
        "requires_escalation": False,
    },
    {
        "text": "I can't log in. It keeps saying 'Invalid credentials' even though I reset my password.",
        "true_category": Category.TECHNICAL,
        "true_priority": Priority.HIGH,
        "requires_escalation": False,
    },
    {
        "text": "The API is returning 500 errors intermittently. Here's the error: Internal Server Error on /v2/data.",
        "true_category": Category.TECHNICAL,
        "true_priority": Priority.HIGH,
        "requires_escalation": True,
    },
    {
        "text": "How do I integrate your platform with Zapier? I followed the docs but the webhook isn't firing.",
        "true_category": Category.TECHNICAL,
        "true_priority": Priority.MEDIUM,
        "requires_escalation": False,
    },
    {
        "text": "Your mobile app on iOS 17 shows a blank white screen after the splash screen.",
        "true_category": Category.TECHNICAL,
        "true_priority": Priority.HIGH,
        "requires_escalation": False,
    },

    # ── GENERAL ──────────────────────────────────────────────────────────────
    {
        "text": "How do I reset my password? I didn't receive the reset email.",
        "true_category": Category.GENERAL,
        "true_priority": Priority.LOW,
        "requires_escalation": False,
    },
    {
        "text": "What are your support hours? I need to know when I can reach someone.",
        "true_category": Category.GENERAL,
        "true_priority": Priority.LOW,
        "requires_escalation": False,
    },
    {
        "text": "Can I have multiple users under one account? How do I add a team member?",
        "true_category": Category.GENERAL,
        "true_priority": Priority.LOW,
        "requires_escalation": False,
    },
    {
        "text": "Do you offer a student discount? I'm currently enrolled at university.",
        "true_category": Category.GENERAL,
        "true_priority": Priority.LOW,
        "requires_escalation": False,
    },
    {
        "text": "I'd like to request a feature: dark mode for the mobile app. Is this on the roadmap?",
        "true_category": Category.GENERAL,
        "true_priority": Priority.LOW,
        "requires_escalation": False,
    },

    # ── COMPLAINT ────────────────────────────────────────────────────────────
    {
        "text": "This is absolutely unacceptable! My data was deleted without warning. I'm furious and considering legal action.",
        "true_category": Category.COMPLAINT,
        "true_priority": Priority.HIGH,
        "requires_escalation": True,
    },
    {
        "text": "Your support team has been ignoring my emails for 5 days. This is a disgrace.",
        "true_category": Category.COMPLAINT,
        "true_priority": Priority.HIGH,
        "requires_escalation": True,
    },
    {
        "text": "I'm extremely disappointed with the service. Nothing works as advertised.",
        "true_category": Category.COMPLAINT,
        "true_priority": Priority.MEDIUM,
        "requires_escalation": True,
    },
    {
        "text": "Your last update broke features that were working fine before. I'm so angry right now.",
        "true_category": Category.COMPLAINT,
        "true_priority": Priority.HIGH,
        "requires_escalation": True,
    },
    {
        "text": "I've been a paying customer for 2 years and this is how you treat loyal customers? Shameful.",
        "true_category": Category.COMPLAINT,
        "true_priority": Priority.MEDIUM,
        "requires_escalation": True,
    },

    # ── POSITIVE ─────────────────────────────────────────────────────────────
    {
        "text": "Just wanted to say your team is fantastic! The onboarding was smooth and quick.",
        "true_category": Category.POSITIVE,
        "true_priority": Priority.LOW,
        "requires_escalation": False,
    },
    {
        "text": "The new analytics dashboard is incredible. Much better than the previous version!",
        "true_category": Category.POSITIVE,
        "true_priority": Priority.LOW,
        "requires_escalation": False,
    },
    {
        "text": "Your agent Sarah was super helpful and resolved my issue in minutes. Please give her recognition!",
        "true_category": Category.POSITIVE,
        "true_priority": Priority.LOW,
        "requires_escalation": False,
    },
    {
        "text": "I recommended your service to three colleagues. Keep up the great work.",
        "true_category": Category.POSITIVE,
        "true_priority": Priority.LOW,
        "requires_escalation": False,
    },
    {
        "text": "5 stars! The best customer support experience I've had from any SaaS tool. Thank you!",
        "true_category": Category.POSITIVE,
        "true_priority": Priority.LOW,
        "requires_escalation": False,
    },
    {
        "text": "Finally got everything set up thanks to your docs. Super impressed with the product.",
        "true_category": Category.POSITIVE,
        "true_priority": Priority.LOW,
        "requires_escalation": False,
    },
    {
        "text": "Your mobile app is by far the cleanest I've used. Loving the experience so far!",
        "true_category": Category.POSITIVE,
        "true_priority": Priority.LOW,
        "requires_escalation": False,
    },
]


def get_ticket_bank(count: int, seed: int = 42) -> List[Ticket]:
    """
    Sample `count` tickets from the bank, seeded for reproducibility.
    IDs are assigned sequentially starting at 1.
    """
    import random
    rng = random.Random(seed)
    sample = rng.sample(_TICKET_BANK, min(count, len(_TICKET_BANK)))
    return [
        Ticket(id=i + 1, **entry)
        for i, entry in enumerate(sample)
    ]
