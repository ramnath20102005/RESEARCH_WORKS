"""
Unit test for streak update logic in session_manager.py

This test directly calls session.add_evaluation() to verify streak updates
without depending on semantic evaluation or the full orchestrator pipeline.
"""

import sys
import logging
from app.interview.session_manager import InterviewSession

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_streak_updates_unit():
    """Test streak update logic directly."""
    logger.info("=" * 80)
    logger.info("STREAK UPDATE UNIT TEST")
    logger.info("=" * 80)

    # Create a session
    session_id = "test_streak_unit"
    resume_data = {"test": "data"}
    session = InterviewSession(session_id, resume_data)

    # Initial state
    logger.info(f"\nInitial: correct={session.correct_streak}, wrong={session.wrong_streak}")
    assert session.correct_streak == 0, f"Initial correct_streak should be 0, got {session.correct_streak}"
    assert session.wrong_streak == 0, f"Initial wrong_streak should be 0, got {session.wrong_streak}"

    # Q1: Incorrect answer
    logger.info("\n" + "=" * 80)
    logger.info("Q1: Incorrect answer")
    logger.info("=" * 80)
    eval1 = {
        'semantic': {'is_correct': False},
        'question_assessment': {'question_difficulty': 'Easy'}
    }
    session.add_evaluation(eval1)
    logger.info(f"After Q1: correct={session.correct_streak}, wrong={session.wrong_streak}")
    assert session.correct_streak == 0, f"Q1 correct_streak should be 0, got {session.correct_streak}"
    assert session.wrong_streak == 1, f"Q1 wrong_streak should be 1, got {session.wrong_streak}"

    # Q2: Incorrect answer
    logger.info("\n" + "=" * 80)
    logger.info("Q2: Incorrect answer")
    logger.info("=" * 80)
    eval2 = {
        'semantic': {'is_correct': False},
        'question_assessment': {'question_difficulty': 'Easy'}
    }
    session.add_evaluation(eval2)
    logger.info(f"After Q2: correct={session.correct_streak}, wrong={session.wrong_streak}")
    assert session.correct_streak == 0, f"Q2 correct_streak should be 0, got {session.correct_streak}"
    assert session.wrong_streak == 2, f"Q2 wrong_streak should be 2, got {session.wrong_streak}"

    # Q3: Correct answer
    logger.info("\n" + "=" * 80)
    logger.info("Q3: Correct answer")
    logger.info("=" * 80)
    eval3 = {
        'semantic': {'is_correct': True},
        'question_assessment': {'question_difficulty': 'Easy'}
    }
    session.add_evaluation(eval3)
    logger.info(f"After Q3: correct={session.correct_streak}, wrong={session.wrong_streak}")
    assert session.correct_streak == 1, f"Q3 correct_streak should be 1, got {session.correct_streak}"
    assert session.wrong_streak == 0, f"Q3 wrong_streak should be 0, got {session.wrong_streak}"

    # Q4: Correct answer (streak should increase)
    logger.info("\n" + "=" * 80)
    logger.info("Q4: Correct answer (streak increase)")
    logger.info("=" * 80)
    eval4 = {
        'semantic': {'is_correct': True},
        'question_assessment': {'question_difficulty': 'Easy'}
    }
    session.add_evaluation(eval4)
    logger.info(f"After Q4: correct={session.correct_streak}, wrong={session.wrong_streak}")
    assert session.correct_streak == 2, f"Q4 correct_streak should be 2, got {session.correct_streak}"
    assert session.wrong_streak == 0, f"Q4 wrong_streak should be 0, got {session.wrong_streak}"

    # Q5: Incorrect answer (correct streak should reset)
    logger.info("\n" + "=" * 80)
    logger.info("Q5: Incorrect answer (correct streak reset)")
    logger.info("=" * 80)
    eval5 = {
        'semantic': {'is_correct': False},
        'question_assessment': {'question_difficulty': 'Easy'}
    }
    session.add_evaluation(eval5)
    logger.info(f"After Q5: correct={session.correct_streak}, wrong={session.wrong_streak}")
    assert session.correct_streak == 0, f"Q5 correct_streak should be 0, got {session.correct_streak}"
    assert session.wrong_streak == 1, f"Q5 wrong_streak should be 1, got {session.wrong_streak}"

    logger.info("\n" + "=" * 80)
    logger.info("TEST PASSED: Streak updates are correct")
    logger.info("=" * 80)
    return True

if __name__ == "__main__":
    try:
        success = test_streak_updates_unit()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Test failed with exception: {e}", exc_info=True)
        sys.exit(1)
