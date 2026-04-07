"""
smoke_test.py — Quick sanity check for SupportEnv v2.

Tests the core loop end-to-end in under 10 seconds.
Run with: python smoke_test.py
"""

import sys
import os

sys.path.insert(0, os.path.abspath("."))

try:
    from src.environment import SupportEnv
    from src.models import Action, ActionType, Category
    from src.generator import generate_tickets
    from src.graders import grade_easy
    print("✅ Imports successful!")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)


def run_tests():
    print("\n=== 🎯 SupportEnv v2 Logic Verification ===")
    passed = 0
    failed = 0

    # Test 1: Generator seed reproducibility
    try:
        t1 = generate_tickets(5, seed=42)
        t2 = generate_tickets(5, seed=42)
        assert [t.text for t in t1] == [t.text for t in t2]
        print(f"✓ [1] Generator: Seed reproducibility verified ({len(t1)} tickets)")
        passed += 1
    except Exception as e:
        print(f"× [1] Generator failed: {e}")
        failed += 1

    # Test 2: Reset
    try:
        env = SupportEnv(task="easy", seed=42, db_path="/tmp/smoke_test.db")
        obs = env.reset()
        assert len(obs.tickets) == 5
        assert env._episode_id is not None
        print(f"✓ [2] Reset: {len(obs.tickets)} tickets loaded, episode_id={env._episode_id}")
        passed += 1
    except Exception as e:
        print(f"× [2] Reset failed: {e}")
        failed += 1

    # Test 3: Correct classification → +0.30
    try:
        ticket = env._tickets[0]
        action = Action(
            action_type=ActionType.CLASSIFY,
            ticket_id=ticket.id,
            content=ticket.true_category.value,
        )
        obs, reward, done, info = env.step(action)
        assert reward == 0.3, f"Expected 0.3, got {reward}"
        print(f"✓ [3] Reward: Correct classification yielded +{reward} (expected +0.30)")
        passed += 1
    except Exception as e:
        print(f"× [3] Reward logic failed: {e}")
        failed += 1

    # Test 4: Disallowed action → -1.0
    try:
        bad_action = Action(
            action_type=ActionType.RESPOND,
            ticket_id=ticket.id,
            content="Hello",
        )
        obs, reward, done, info = env.step(bad_action)
        assert reward == -1.0, f"Expected -1.0, got {reward}"
        print(f"✓ [4] Gating: Disallowed 'respond' in 'easy' penalized {reward} (expected -1.0)")
        passed += 1
    except Exception as e:
        print(f"× [4] Task gating failed: {e}")
        failed += 1

    # Test 5: Repeated action → -0.5 loop penalty
    try:
        env2 = SupportEnv(task="easy", seed=99, db_path="/tmp/smoke_test2.db")
        env2.reset()
        t = env2._tickets[0]
        env2.step(Action(action_type=ActionType.CLASSIFY, ticket_id=t.id, content="billing"))
        obs, reward, done, info = env2.step(Action(
            action_type=ActionType.CLASSIFY, ticket_id=t.id, content="billing",
        ))
        assert reward == -0.5, f"Expected -0.5, got {reward}"
        print(f"✓ [5] Loop detection: Repeated action on same ticket → {reward} (expected -0.50)")
        passed += 1
    except Exception as e:
        print(f"× [5] Loop detection failed: {e}")
        failed += 1

    # Test 6: Perfect grading → 1.0
    try:
        env3 = SupportEnv(task="easy", seed=0, db_path="/tmp/smoke_test3.db")
        env3.reset()
        for t in env3._tickets:
            env3.step(Action(
                action_type=ActionType.CLASSIFY,
                ticket_id=t.id,
                content=t.true_category.value,
            ))
        report = env3.grade()
        assert report.score == 0.99, f"Expected 0.99, got {report.score}"
        print(f"✓ [6] Grading: Perfect run on 'easy' → {report.score*100:.0f}%")
        passed += 1
    except Exception as e:
        print(f"× [6] Grading failed: {e}")
        failed += 1

    # Test 7: Info dict structure
    try:
        env4 = SupportEnv(task="medium", seed=42, db_path="/tmp/smoke_test4.db")
        env4.reset()
        t = env4._tickets[0]
        obs, reward, done, info = env4.step(Action(
            action_type=ActionType.CLASSIFY, ticket_id=t.id,
            content=t.true_category.value,
        ))
        assert "tickets_resolved" in info
        assert "efficiency" in info
        assert "loops_detected" in info
        assert "priority_misses" in info
        print(f"✓ [7] Info dict: {list(info.keys())}")
        passed += 1
    except Exception as e:
        print(f"× [7] Info dict failed: {e}")
        failed += 1

    # Test 8: Database replay
    try:
        env5 = SupportEnv(task="easy", seed=77, db_path="/tmp/smoke_test5.db")
        env5.reset()
        t = env5._tickets[0]
        env5.step(Action(action_type=ActionType.CLASSIFY, ticket_id=t.id, content="billing"))
        env5.grade()
        ep_id = env5._episode_id
        actions = env5.replay(ep_id)
        assert len(actions) == 1
        print(f"✓ [8] Replay: Episode {ep_id} reconstructed with {len(actions)} action(s)")
        passed += 1
    except Exception as e:
        print(f"× [8] Replay failed: {e}")
        failed += 1

    print(f"\n=== Results: {passed} passed, {failed} failed ===")
    if failed == 0:
        print("🚀 All core logic verified! SupportEnv v2 is ready.")
    else:
        print("⚠️  Some tests failed — see output above.")
    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
