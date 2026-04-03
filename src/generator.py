"""
generator.py — Procedural, seed-based ticket generator for SupportEnv.

Replaces the static ticket bank with a fully dynamic, reproducible generator.
Supports:
  - Seed-controlled randomness (identical seed → identical tickets)
  - 5 categories: billing, technical, general, complaint, positive
  - 3 user personas: angry, polite, confused
  - Noise injection: typos, random casing, punctuation errors
  - Rich metadata: required_keywords, forbidden_phrases, resolution_steps
"""

from __future__ import annotations

import random
import re
from typing import List, Optional

from src.models import Category, Priority, Ticket


# ─────────────────────────────────────────────────────────────────────────────
# Template Banks
# ─────────────────────────────────────────────────────────────────────────────

_TEMPLATES: dict[Category, list[dict]] = {
    Category.BILLING: [
        {
            "text": "I was charged {amount} twice this month. Please refund one of the payments immediately.",
            "priority": Priority.HIGH,
            "requires_escalation": False,
            "required_keywords": ["refund", "billing", "charge"],
            "forbidden_phrases": ["ignore this", "not my problem"],
            "resolution_steps": ["Verify duplicate charge", "Issue refund within 3-5 business days", "Send confirmation email"],
        },
        {
            "text": "My invoice #{invoice_id} shows a charge of {amount} I don't recognize. Can you explain this?",
            "priority": Priority.MEDIUM,
            "requires_escalation": False,
            "required_keywords": ["invoice", "charge", "billing"],
            "forbidden_phrases": ["refuse", "won't help"],
            "resolution_steps": ["Pull invoice #{invoice_id}", "Explain charge details", "Offer adjustment if erroneous"],
        },
        {
            "text": "I cancelled my {plan} plan {days} days ago but was still billed {amount}. I want a full refund or I'll dispute the charge.",
            "priority": Priority.HIGH,
            "requires_escalation": True,
            "required_keywords": ["refund", "cancel", "billing", "subscription"],
            "forbidden_phrases": ["non-refundable", "sorry can't help"],
            "resolution_steps": ["Confirm cancellation date", "Verify billing cycle", "Issue prorated refund", "Escalate to finance if over $100"],
        },
        {
            "text": "How do I update my payment method from a debit card to a credit card?",
            "priority": Priority.LOW,
            "requires_escalation": False,
            "required_keywords": ["payment", "billing", "update"],
            "forbidden_phrases": ["not possible", "can't do that"],
            "resolution_steps": ["Navigate to Settings > Billing", "Click 'Update Payment Method'", "Enter new card details"],
        },
        {
            "text": "I'd like to upgrade from {plan} to {plan2}. What's the prorated cost for the rest of the month?",
            "priority": Priority.LOW,
            "requires_escalation": False,
            "required_keywords": ["billing", "subscription", "cost", "upgrade"],
            "forbidden_phrases": ["no discounts", "full price only"],
            "resolution_steps": ["Calculate prorated amount", "Apply upgrade in dashboard", "Confirm new billing cycle"],
        },
        {
            "text": "I received a {amount} charge on {date} that I did not authorize. This is fraudulent activity.",
            "priority": Priority.HIGH,
            "requires_escalation": True,
            "required_keywords": ["refund", "billing", "charge", "unauthorized"],
            "forbidden_phrases": ["your problem", "can't help"],
            "resolution_steps": ["Flag as unauthorized charge", "Initiate chargeback review", "Escalate to fraud team", "Notify customer within 24h"],
        },
    ],
    Category.TECHNICAL: [
        {
            "text": "The app crashes every time I try to export my report to {format}. This is blocking my work.",
            "priority": Priority.HIGH,
            "requires_escalation": False,
            "required_keywords": ["fix", "bug", "error", "issue"],
            "forbidden_phrases": ["works on my end", "not our fault"],
            "resolution_steps": ["Reproduce crash in staging", "Check export logs for stack trace", "Deploy hotfix or provide workaround"],
        },
        {
            "text": "The dashboard takes over {seconds} seconds to load on every page refresh.",
            "priority": Priority.MEDIUM,
            "requires_escalation": False,
            "required_keywords": ["fix", "slow", "issue", "performance"],
            "forbidden_phrases": ["just wait", "use a faster computer"],
            "resolution_steps": ["Check CDN performance metrics", "Identify slow API calls", "Enable caching if disabled"],
        },
        {
            "text": "I can't log in. It keeps saying 'Invalid credentials' even after resetting my password {retries} times.",
            "priority": Priority.HIGH,
            "requires_escalation": False,
            "required_keywords": ["login", "password", "reset", "fix"],
            "forbidden_phrases": ["forget it", "can't help with that"],
            "resolution_steps": ["Manually reset account from admin panel", "Clear cached credentials", "Verify email confirmation"],
        },
        {
            "text": "The API is returning {error_code} errors on /v2/{endpoint}. This has been happening for {hours} hours.",
            "priority": Priority.HIGH,
            "requires_escalation": True,
            "required_keywords": ["api", "error", "fix", "bug"],
            "forbidden_phrases": ["not our api", "check your code"],
            "resolution_steps": ["Check API gateway logs", "Identify affected endpoints", "Roll back last deployment if needed", "Update status page"],
        },
        {
            "text": "Your mobile app on {os} shows a blank white screen after the splash screen.",
            "priority": Priority.HIGH,
            "requires_escalation": False,
            "required_keywords": ["fix", "bug", "crash", "issue"],
            "forbidden_phrases": ["buy a new phone", "update manually"],
            "resolution_steps": ["Reproduce on test device", "Check for OS-specific rendering bug", "Push hotfix to app store"],
        },
        {
            "text": "The Zapier integration stopped working after your {date} update. Webhooks aren't firing.",
            "priority": Priority.MEDIUM,
            "requires_escalation": False,
            "required_keywords": ["api", "fix", "error", "integration"],
            "forbidden_phrases": ["use a different tool", "not supported"],
            "resolution_steps": ["Test webhook endpoint manually", "Check Zapier logs", "Verify API key permissions", "Provide updated integration guide"],
        },
    ],
    Category.GENERAL: [
        {
            "text": "How do I reset my password? I didn't receive the reset email after {minutes} minutes.",
            "priority": Priority.LOW,
            "requires_escalation": False,
            "required_keywords": ["how", "reset", "help"],
            "forbidden_phrases": ["figure it out yourself", "rtfm"],
            "resolution_steps": ["Check spam folder", "Resend reset email from admin", "Verify email address on file"],
        },
        {
            "text": "What are your support hours? I need to reach someone for a {topic} question.",
            "priority": Priority.LOW,
            "requires_escalation": False,
            "required_keywords": ["how", "when", "help", "team"],
            "forbidden_phrases": ["we don't support that", "no idea"],
            "resolution_steps": ["Provide support hours (Mon-Fri 9am-6pm UTC)", "Share help center link", "Offer async ticket option"],
        },
        {
            "text": "Can I have multiple users under one account? I need to add {count} team members.",
            "priority": Priority.LOW,
            "requires_escalation": False,
            "required_keywords": ["how", "team", "add", "help"],
            "forbidden_phrases": ["no you can't", "pay more"],
            "resolution_steps": ["Navigate to Settings > Team Members", "Send invites via email", "Assign roles as needed"],
        },
        {
            "text": "Do you offer any discount for {group} users? I'm currently {affiliation}.",
            "priority": Priority.LOW,
            "requires_escalation": False,
            "required_keywords": ["discount", "how", "help"],
            "forbidden_phrases": ["no discounts ever", "full price only"],
            "resolution_steps": ["Check discount eligibility", "Apply promo code if applicable", "Escalate to sales if bulk deal"],
        },
        {
            "text": "I'd like to request a feature: {feature}. Is this on the roadmap?",
            "priority": Priority.LOW,
            "requires_escalation": False,
            "required_keywords": ["feature", "update", "when", "add"],
            "forbidden_phrases": ["we don't take requests", "not possible"],
            "resolution_steps": ["Log feature request internally", "Check public roadmap", "Notify user when shipped"],
        },
    ],
    Category.COMPLAINT: [
        {
            "text": "This is absolutely unacceptable! My {data_type} data was deleted without warning. I'm considering legal action.",
            "priority": Priority.HIGH,
            "requires_escalation": True,
            "required_keywords": ["sorry", "apologies", "escalate", "resolve", "priority"],
            "forbidden_phrases": ["your fault", "that's fine", "ignore"],
            "resolution_steps": ["Escalate to senior support", "Attempt data recovery", "Send formal apology", "Offer service credit"],
        },
        {
            "text": "Your support team has been ignoring my emails for {days} days. This is completely unacceptable.",
            "priority": Priority.HIGH,
            "requires_escalation": True,
            "required_keywords": ["sorry", "apologies", "priority", "manager", "resolve"],
            "forbidden_phrases": ["we're busy", "wait longer"],
            "resolution_steps": ["Immediately assign dedicated agent", "Send acknowledgment within 1 hour", "Escalate to manager"],
        },
        {
            "text": "I'm extremely disappointed with {feature}. It doesn't work as advertised and I've wasted {hours} hours trying to make it work.",
            "priority": Priority.MEDIUM,
            "requires_escalation": True,
            "required_keywords": ["sorry", "apologies", "understand", "frustration", "resolve"],
            "forbidden_phrases": ["not our problem", "works as intended"],
            "resolution_steps": ["Acknowledge frustration", "Provide guided walkthrough", "Offer extended trial or credit"],
        },
        {
            "text": "Your last {version} update broke {feature} that was working perfectly before. I'm furious.",
            "priority": Priority.HIGH,
            "requires_escalation": True,
            "required_keywords": ["sorry", "fix", "priority", "resolve", "escalate"],
            "forbidden_phrases": ["expected behavior", "upgrade your plan"],
            "resolution_steps": ["Confirm regression bug", "Roll back for affected user", "Expedite fix in next patch"],
        },
        {
            "text": "I've been a paying customer for {years} years and this is how you treat loyal customers? Absolutely shameful.",
            "priority": Priority.MEDIUM,
            "requires_escalation": True,
            "required_keywords": ["sorry", "apologies", "compensate", "priority", "resolve"],
            "forbidden_phrases": ["everyone has issues", "nothing we can do"],
            "resolution_steps": ["Acknowledge loyalty", "Offer retention credit", "Escalate to account manager"],
        },
    ],
    Category.POSITIVE: [
        {
            "text": "Just wanted to say your team is fantastic! The {feature} onboarding was smooth and quick.",
            "priority": Priority.LOW,
            "requires_escalation": False,
            "required_keywords": ["thank", "glad", "appreciate", "feedback"],
            "forbidden_phrases": ["no response needed", "don't care"],
            "resolution_steps": ["Thank customer", "Share feedback with team", "Invite to leave public review"],
        },
        {
            "text": "The new {feature} dashboard is incredible. Much better than the previous version!",
            "priority": Priority.LOW,
            "requires_escalation": False,
            "required_keywords": ["thank", "happy", "glad", "appreciate"],
            "forbidden_phrases": ["whatever", "ok sure"],
            "resolution_steps": ["Express gratitude", "Note positive feedback in CRM", "Share with product team"],
        },
        {
            "text": "Your agent {name} was super helpful and resolved my issue in {minutes} minutes. Please give them recognition!",
            "priority": Priority.LOW,
            "requires_escalation": False,
            "required_keywords": ["thank", "appreciate", "feedback", "noted"],
            "forbidden_phrases": ["irrelevant", "not our concern"],
            "resolution_steps": ["Thank customer", "Forward praise to agent", "Add to agent performance record"],
        },
        {
            "text": "I recommended your service to {count} colleagues this week. Keep up the great work!",
            "priority": Priority.LOW,
            "requires_escalation": False,
            "required_keywords": ["thank", "glad", "appreciate", "feedback"],
            "forbidden_phrases": ["we don't care", "so what"],
            "resolution_steps": ["Thank customer", "Offer referral bonus if program exists", "Log for NPS metrics"],
        },
        {
            "text": "{stars} stars! Best customer support experience from any SaaS tool. Thank you!",
            "priority": Priority.LOW,
            "requires_escalation": False,
            "required_keywords": ["thank", "welcome", "appreciate", "glad"],
            "forbidden_phrases": ["ok", "whatever"],
            "resolution_steps": ["Express gratitude", "Request public review", "Share with leadership"],
        },
    ],
}

# Template variable pools
_VARS: dict[str, list] = {
    "amount":     ["$19.99", "$49.99", "$99.00", "$149.00", "$9.99"],
    "invoice_id": ["4872", "10391", "88201", "3314", "55098"],
    "plan":       ["Basic", "Starter", "Professional", "Enterprise"],
    "plan2":      ["Pro", "Business", "Enterprise", "Team"],
    "days":       ["3", "7", "14", "2", "10"],
    "date":       ["January 15", "February 3", "March 22", "April 1"],
    "format":     ["PDF", "CSV", "Excel", "JSON"],
    "seconds":    ["30", "45", "60", "90"],
    "retries":    ["3", "5", "2", "4"],
    "error_code": ["500", "503", "502", "429"],
    "endpoint":   ["data", "reports", "users", "analytics"],
    "hours":      ["2", "4", "6", "12", "24"],
    "os":         ["iOS 17", "Android 14", "iOS 16", "Android 13"],
    "minutes":    ["10", "15", "30", "5", "20"],
    "topic":      ["billing", "technical", "account", "integration"],
    "count":      ["3", "5", "10", "2", "8"],
    "group":      ["student", "nonprofit", "startup", "educational"],
    "affiliation":["enrolled at university", "a non-profit org", "an early-stage startup"],
    "feature":    ["dark mode", "API v3", "team collaboration", "export", "mobile app", "Zapier integration", "SSO"],
    "data_type":  ["project", "report", "customer", "analytics", "account"],
    "version":    ["v2.4", "v3.0", "v2.8", "v3.1"],
    "years":      ["2", "3", "4", "1"],
    "name":       ["Sarah", "Alex", "Jordan", "Sam", "Chris"],
    "stars":      ["5", "4"],
}

# Persona modifiers
_PERSONA_PREFIXES: dict[str, list[str]] = {
    "angry": [
        "This is TOTALLY unacceptable!! ",
        "I'm so frustrated right now. ",
        "Unbelievable - ",
        "I NEED this fixed NOW. ",
        "I'm about to cancel if ",
    ],
    "polite": [
        "Hi there, hope you're doing well! ",
        "Hello, I wanted to reach out because ",
        "Good day, I'm hoping you can help me with ",
        "Hi team, quick question: ",
        "Thanks for being available. ",
    ],
    "confused": [
        "Hi I'm not sure if I'm doing this right but ",
        "Um so I think maybe ",
        "Sorry if this is obvious but ",
        "I might be missing something but ",
        "Not sure who to contact but ",
    ],
}

_PERSONA_SUFFIXES: dict[str, list[str]] = {
    "angry": [" Fix this NOW.", " This is ridiculous.", " I expect an immediate response.", " Absolutely unacceptable."],
    "polite": [" Thank you so much!", " Appreciate your help.", " Thanks in advance!", " Have a great day."],
    "confused": [" Hope that makes sense?", " Let me know if you need more info.", " Not sure if I explained that right.", ""],
}

# Noise functions
_TYPO_MAP = {
    "the": "teh", "is": "si", "your": "yoru", "please": "pelase",
    "account": "accont", "payment": "payemnt", "issue": "isuue",
    "problem": "probelm", "help": "hlep", "error": "eror",
}


def _inject_noise(text: str, rng: random.Random, noise_level: float = 0.15) -> str:
    """Inject realistic noise (typos, casing) with probability noise_level."""
    words = text.split()
    result = []
    for word in words:
        if rng.random() < noise_level:
            clean = word.lower().strip(".,!?")
            if clean in _TYPO_MAP:
                # Preserve trailing punctuation
                trail = word[len(clean):]
                word = _TYPO_MAP[clean] + trail
        result.append(word)
    # Random casing on ~10% of words
    if rng.random() < 0.2:
        idx = rng.randint(0, len(result) - 1)
        result[idx] = result[idx].upper()
    return " ".join(result)


def _fill_template(template: str, rng: random.Random) -> str:
    """Replace {var} placeholders with random values from _VARS."""
    def replacer(match):
        key = match.group(1)
        choices = _VARS.get(key, [f"<{key}>"])
        return rng.choice(choices)
    return re.sub(r"\{(\w+)\}", replacer, template)


# ─────────────────────────────────────────────────────────────────────────────
# Public Generator
# ─────────────────────────────────────────────────────────────────────────────

_CATEGORIES_ORDERED = [
    Category.BILLING,
    Category.TECHNICAL,
    Category.GENERAL,
    Category.COMPLAINT,
    Category.POSITIVE,
]

_PERSONAS = ["angry", "polite", "confused"]


def generate_tickets(count: int, seed: int = 42) -> List[Ticket]:
    """
    Generate `count` tickets procedurally using the given seed.

    Guarantees:
      - Same seed → same tickets (fully reproducible)
      - Category distribution is balanced (round-robin across 5 categories)
      - Each ticket has required_keywords, forbidden_phrases, resolution_steps
      - Persona and noise are applied consistently per seed
    """
    rng = random.Random(seed)
    tickets: List[Ticket] = []

    for i in range(count):
        # Round-robin category selection ensures balanced distribution
        category = _CATEGORIES_ORDERED[i % len(_CATEGORIES_ORDERED)]
        templates = _TEMPLATES[category]
        template_dict = rng.choice(templates)

        # Fill template variables
        raw_text = _fill_template(template_dict["text"], rng)

        # Assign persona
        persona = rng.choice(_PERSONAS)

        # Build personalized text
        prefix = rng.choice(_PERSONA_PREFIXES[persona])
        suffix = rng.choice(_PERSONA_SUFFIXES[persona])

        # Capitalize first char of raw_text if prefix adds one
        if prefix.endswith(("because ", "with ", "if ", "but ")):
            raw_text = raw_text[0].lower() + raw_text[1:]

        full_text = prefix + raw_text + suffix

        # Inject noise (angry persona gets less noise, confused gets more)
        noise_levels = {"angry": 0.05, "polite": 0.08, "confused": 0.18}
        full_text = _inject_noise(full_text, rng, noise_levels[persona])

        ticket = Ticket(
            id=i + 1,
            text=full_text,
            true_category=category,
            true_priority=template_dict["priority"],
            requires_escalation=template_dict["requires_escalation"],
            required_keywords=template_dict["required_keywords"],
            forbidden_phrases=template_dict["forbidden_phrases"],
            resolution_steps=template_dict["resolution_steps"],
            persona=persona,
            dependencies=[],       # populated by dynamics.py
            escalation_step=rng.randint(5, 15),  # dynamic escalation threshold
        )
        tickets.append(ticket)

    return tickets
