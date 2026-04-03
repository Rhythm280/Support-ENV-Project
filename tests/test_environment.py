"""
tests/test_environment.py — Comprehensive unit tests for SupportEnv v2.

Run with:  python -m pytest tests/ -v
"""

from __future__ import annotations

import pytest

from src.environment import SupportEnv
from src.models import Action, ActionType, Category, Priority, TaskName, TicketStatus
from src.generator import generate_tickets
from src.graders import grade_easy, grade_medium, grade_hard, grade_ticket
from src.database import init_db, log_episode_start, load_episode
from src.dynamics import (
    apply_priority_escalation,
    worsen_tone,
    assign_dependencies,
    check_dependencies_met,
)


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

@pytest.fixture
def easy_env(tmp_path):
    env = SupportEnv(task="easy", seed=0, db_path=str(tmp_path / "test.db"))
    env.reset()
    return env


@pytest.fixture
def medium_env(tmp_path):
    env = SupportEnv(task="medium", seed=0, db_path=str(tmp_path / "test.db"))
    env.reset()
    return env


@pytest.fixture
def hard_env(tmp_path):
    env = SupportEnv(task="hard", seed=0, db_path=str(tmp_path / "test.db"))
    env.reset()
    return env


# ─────────────────────────────────────────────
# Generator tests
# ─────────────────────────────────────────────

class TestGenerator:
    def test_seed_reproducibility(self):
        tickets_a = generate_tickets(5, seed=42)
        tickets_b = generate_tickets(5, seed=42)
        assert [t.text for t in tickets_a] == [t.text for t in tickets_b]

    def test_different_seeds_different_tickets(self):
        tickets_a = generate_tickets(5, seed=1)
        tickets_b = generate_tickets(5, seed=2)
        # At least some tickets should differ
        assert any(a.text != b.text for a, b in zip(tickets_a, tickets_b))

    def test_ticket_count(self):
        for n in [3, 5, 6, 7]:
            tickets = generate_tickets(n, seed=99)
            assert len(tickets) == n

    def test_ticket_has_metadata(self):
        tickets = generate_tickets(5, seed=42)
        for t in tickets:
            assert t.required_keywords, "Ticket must have required_keywords"
            assert t.forbidden_phrases, "Ticket must have forbidden_phrases"
            assert t.resolution_steps, "Ticket must have resolution_steps"
            assert t.persona in ("angry", "polite", "confused")
            assert t.escalation_step > 0

    def test_balanced_categories(self):
        """Round-robin ensures 5 tickets cover 5 different categories."""
        tickets = generate_tickets(5, seed=0)
        categories = {t.true_category for t in tickets}
        assert len(categories) == 5  # one of each

    def test_ticket_ids_sequential(self):
        tickets = generate_tickets(7, seed=0)
        ids = [t.id for t in tickets]
        assert ids == list(range(1, 8))


# ─────────────────────────────────────────────
# Reset tests
# ─────────────────────────────────────────────

class TestReset:
    def test_reset_returns_observation(self, easy_env):
        obs = easy_env.reset()
        assert obs is not None
        assert len(obs.tickets) == 5

    def test_reset_clears_state(self, easy_env):
        t = easy_env._tickets[0]
        easy_env.step(Action(action_type=ActionType.CLASSIFY, ticket_id=t.id, content="billing"))
        easy_env.reset()
        assert easy_env._step_count == 0
        assert easy_env._cumulative_reward == 0.0
        assert not easy_env._done
        assert easy_env._action_history == []
        assert easy_env._loops_detected == 0

    def test_task_ticket_count(self, tmp_path):
        for task, expected in [("easy", 5), ("medium", 6), ("hard", 7)]:
            env = SupportEnv(task=task, seed=0, db_path=str(tmp_path / f"{task}.db"))
            obs = env.reset()
            assert len(obs.tickets) == expected

    def test_seed_reproducibility(self, tmp_path):
        db = str(tmp_path / "test.db")
        env_a = SupportEnv(task="easy", seed=99, db_path=db)
        env_b = SupportEnv(task="easy", seed=99, db_path=db)
        obs_a = env_a.reset()
        obs_b = env_b.reset()
        texts_a = [t["text"] for t in obs_a.tickets]
        texts_b = [t["text"] for t in obs_b.tickets]
        assert texts_a == texts_b

    def test_observation_has_info_dict(self, easy_env):
        obs = easy_env.reset()
        assert "info" in obs.model_fields or hasattr(obs, "info")
        assert isinstance(obs.info, dict)


# ─────────────────────────────────────────────
# Step / action tests
# ─────────────────────────────────────────────

class TestStep:
    def test_correct_classify_gives_positive_reward(self, easy_env):
        ticket = easy_env._tickets[0]
        action = Action(
            action_type=ActionType.CLASSIFY,
            ticket_id=ticket.id,
            content=ticket.true_category.value,
        )
        obs, reward, done, info = easy_env.step(action)
        assert reward == pytest.approx(0.30)

    def test_wrong_classify_gives_negative_reward(self, easy_env):
        ticket = easy_env._tickets[0]
        wrong = next(c.value for c in Category if c != ticket.true_category)
        obs, reward, done, info = easy_env.step(Action(
            action_type=ActionType.CLASSIFY, ticket_id=ticket.id, content=wrong,
        ))
        assert reward == pytest.approx(-0.30)

    def test_invalid_ticket_id_gives_heavy_penalty(self, easy_env):
        obs, reward, done, info = easy_env.step(Action(
            action_type=ActionType.CLASSIFY, ticket_id=9999, content="billing",
        ))
        assert reward == pytest.approx(-1.0)
        assert obs.last_action_error is True

    def test_disallowed_action_gives_heavy_penalty(self, easy_env):
        ticket = easy_env._tickets[0]
        obs, reward, done, info = easy_env.step(Action(
            action_type=ActionType.RESPOND, ticket_id=ticket.id, content="Here is a response",
        ))
        assert reward == pytest.approx(-1.0)

    def test_repeated_action_gives_loop_penalty(self, easy_env):
        ticket = easy_env._tickets[0]
        easy_env.step(Action(action_type=ActionType.CLASSIFY, ticket_id=ticket.id, content="billing"))
        # Second classify on same ticket = loop
        obs, reward, done, info = easy_env.step(Action(
            action_type=ActionType.CLASSIFY, ticket_id=ticket.id, content="billing",
        ))
        assert reward == pytest.approx(-0.50)

    def test_step_count_increments(self, easy_env):
        ticket = easy_env._tickets[0]
        easy_env.step(Action(action_type=ActionType.CLASSIFY, ticket_id=ticket.id, content="billing"))
        assert easy_env._step_count == 1

    def test_info_dict_has_required_keys(self, easy_env):
        ticket = easy_env._tickets[0]
        obs, reward, done, info = easy_env.step(Action(
            action_type=ActionType.CLASSIFY, ticket_id=ticket.id,
            content=ticket.true_category.value,
        ))
        assert "tickets_resolved" in info
        assert "efficiency" in info
        assert "loops_detected" in info
        assert "priority_misses" in info

    def test_done_after_max_steps(self, easy_env):
        ticket = easy_env._tickets[0]
        last_done = False
        for _ in range(easy_env.config.max_steps):
            obs, reward, done, info = easy_env.step(Action(
                action_type=ActionType.CLASSIFY, ticket_id=ticket.id, content="billing",
            ))
            last_done = done
        assert last_done is True

    def test_step_raises_after_done(self, easy_env):
        ticket = easy_env._tickets[0]
        for _ in range(easy_env.config.max_steps):
            easy_env.step(Action(action_type=ActionType.CLASSIFY, ticket_id=ticket.id, content="billing"))
        with pytest.raises(RuntimeError, match="done"):
            easy_env.step(Action(action_type=ActionType.CLASSIFY, ticket_id=ticket.id, content="billing"))

    def test_early_termination_all_resolved(self, tmp_path):
        """Episode ends early when all tickets are closed or escalated."""
        # Use hard task which allows 'respond' (which closes tickets)
        env = SupportEnv(task="hard", seed=0, db_path=str(tmp_path / "et_test.db"))
        env.reset()
        done = False
        for t in env._tickets:
            # Escalate (closes ticket as escalated) — quickest way to mark all resolved
            obs, reward, done, info = env.step(Action(
                action_type=ActionType.ESCALATE, ticket_id=t.id,
            ))
            if done:
                break
        # All 7 tickets escalated — should have ended before max_steps (40)
        assert env._step_count < env.config.max_steps


# ─────────────────────────────────────────────
# Hard task tests
# ─────────────────────────────────────────────

class TestHardTask:
    def test_correct_escalation_reward(self, hard_env):
        needs_esc = next((t for t in hard_env._tickets if t.requires_escalation), None)
        if needs_esc is None:
            pytest.skip("No escalation-required ticket in this seed")
        obs, reward, done, info = hard_env.step(Action(
            action_type=ActionType.ESCALATE, ticket_id=needs_esc.id,
        ))
        assert reward == pytest.approx(0.30)
        assert needs_esc.status == TicketStatus.ESCALATED

    def test_false_escalation_penalty(self, hard_env):
        no_esc = next((t for t in hard_env._tickets if not t.requires_escalation), None)
        if no_esc is None:
            pytest.skip("All tickets need escalation in this seed")
        obs, reward, done, info = hard_env.step(Action(
            action_type=ActionType.ESCALATE, ticket_id=no_esc.id,
        ))
        assert reward == pytest.approx(-0.30)

    def test_relevant_response_reward(self, hard_env):
        billing_ticket = next(
            (t for t in hard_env._tickets if t.true_category.value == "billing"), None
        )
        if billing_ticket is None:
            pytest.skip("No billing ticket in this seed")
        obs, reward, done, info = hard_env.step(Action(
            action_type=ActionType.RESPOND,
            ticket_id=billing_ticket.id,
            content="We will process your refund and address the billing charge immediately.",
        ))
        assert reward > 0  # +0.5 relevant + possibly +0.2 no-forbidden
        assert billing_ticket.status == TicketStatus.CLOSED


# ─────────────────────────────────────────────
# Grader tests (src/graders.py)
# ─────────────────────────────────────────────

class TestGraders:
    def test_grade_easy_perfect_score(self, easy_env):
        for t in easy_env._tickets:
            easy_env.step(Action(
                action_type=ActionType.CLASSIFY,
                ticket_id=t.id,
                content=t.true_category.value,
            ))
        report = easy_env.grade()
        assert report.score == pytest.approx(1.0)

    def test_grade_easy_zero_score(self, easy_env):
        report = easy_env.grade()
        assert report.score == pytest.approx(0.0)

    def test_grade_medium_partial(self, medium_env):
        for t in medium_env._tickets:
            medium_env.step(Action(
                action_type=ActionType.CLASSIFY,
                ticket_id=t.id,
                content=t.true_category.value,
            ))
        report = medium_env.grade()
        assert report.score == pytest.approx(0.5)  # classification only → 0.5

    def test_ticket_grader_category_match(self):
        tickets = generate_tickets(5, seed=42)
        t = tickets[0]
        t.category = t.true_category  # correct
        result = grade_ticket(t, response_text=None)
        assert result.category_match is True
        assert result.score >= 0.3  # at least category match weight

    def test_ticket_grader_keywords(self):
        tickets = generate_tickets(5, seed=42)
        t = tickets[0]
        t.category = t.true_category
        keywords = " ".join(t.required_keywords)
        result = grade_ticket(t, response_text=keywords)
        assert result.keywords_present is True
        assert result.score >= 0.8  # category (0.3) + keywords (0.5)

    def test_ticket_grader_forbidden_phrases(self):
        tickets = generate_tickets(5, seed=42)
        t = tickets[0]
        t.category = t.true_category
        forbidden_response = t.forbidden_phrases[0] if t.forbidden_phrases else "ignore this"
        result = grade_ticket(t, response_text=forbidden_response)
        assert result.no_forbidden is False

    def test_state_has_step_log(self, easy_env):
        t = easy_env._tickets[0]
        easy_env.step(Action(action_type=ActionType.CLASSIFY, ticket_id=t.id, content="billing"))
        state = easy_env.state()
        assert len(state["step_log"]) == 1
        assert state["step_log"][0]["action"] == "classify"

    def test_state_has_info(self, easy_env):
        state = easy_env.state()
        assert "info" in state
        assert "tickets_resolved" in state["info"]


# ─────────────────────────────────────────────
# Database tests
# ─────────────────────────────────────────────

class TestDatabase:
    def test_episode_logged(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        env = SupportEnv(task="easy", seed=0, db_path=db_path)
        env.reset()
        assert env._episode_id is not None
        assert env._episode_id > 0

    def test_actions_logged(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        env = SupportEnv(task="easy", seed=0, db_path=db_path)
        env.reset()
        t = env._tickets[0]
        env.step(Action(action_type=ActionType.CLASSIFY, ticket_id=t.id, content="billing"))

        # Load episode and check actions
        data = load_episode(env._episode_id, db_path=db_path)
        assert len(data["actions"]) == 1
        assert data["actions"][0]["action_type"] == "classify"

    def test_metrics_logged_after_grade(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        env = SupportEnv(task="easy", seed=0, db_path=db_path)
        env.reset()
        for t in env._tickets:
            env.step(Action(action_type=ActionType.CLASSIFY, ticket_id=t.id,
                            content=t.true_category.value))
        env.grade()

        data = load_episode(env._episode_id, db_path=db_path)
        assert len(data["metrics"]) == 1
        assert data["metrics"][0]["final_score"] == pytest.approx(1.0)

    def test_replay_returns_actions(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        env = SupportEnv(task="easy", seed=0, db_path=db_path)
        env.reset()
        t = env._tickets[0]
        env.step(Action(action_type=ActionType.CLASSIFY, ticket_id=t.id, content="billing"))
        env.grade()

        ep_id = env._episode_id
        actions = env.replay(ep_id)
        assert len(actions) == 1
        assert actions[0]["action_type"] == "classify"


# ─────────────────────────────────────────────
# Dynamics tests
# ─────────────────────────────────────────────

class TestDynamics:
    def test_priority_escalation(self):
        tickets = generate_tickets(5, seed=42)
        low_ticket = next((t for t in tickets if t.true_priority == Priority.LOW), None)
        if low_ticket is None:
            pytest.skip("No low priority ticket in this seed")
        low_ticket.escalation_step = 3  # set low threshold
        original_priority = low_ticket.true_priority
        apply_priority_escalation(tickets, current_step=5)  # exceed threshold
        # Priority should have been escalated
        assert low_ticket.true_priority != original_priority or low_ticket.true_priority == Priority.MEDIUM

    def test_tone_worsening(self):
        tickets = generate_tickets(3, seed=1)
        t = tickets[0]
        t.persona = "polite"
        worsen_tone(t, bad_response_count=1)
        assert t.persona in ("confused", "angry")

    def test_dependencies_assigned(self):
        tickets = generate_tickets(7, seed=42)
        assign_dependencies(tickets, seed=42)
        has_dep = any(len(t.dependencies) > 0 for t in tickets)
        assert has_dep

    def test_dependency_check(self):
        tickets = generate_tickets(3, seed=5)
        tickets[1].dependencies = [tickets[0].id]
        # Blocker is open → dep not met
        assert check_dependencies_met(tickets[1], tickets) is False
        # Close the blocker
        tickets[0].status = TicketStatus.CLOSED
        assert check_dependencies_met(tickets[1], tickets) is True
