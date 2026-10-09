"""
Streak update tracing test.

This test simulates the exact three-answer sequence to trace streak updates:
- Q1: Non-answer (incorrect)
- Q2: Non-answer (incorrect)
- Q3: Correct answer

Expected progression:
- Q1: correct=0, wrong=1
- Q2: correct=0, wrong=2
- Q3: correct=1, wrong=0

This test logs streak state at every stage to identify where doubling occurs.
"""

import sys
import logging
from app.interview.adaptive_orchestrator import AdaptiveInterviewOrchestrator
from app.interview.session_manager import InterviewSession

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_streak_updates():
    """Test streak updates through three answers."""
    logger.info("=" * 80)
    logger.info("STREAK UPDATE TRACE TEST")
    logger.info("=" * 80)

    # Initialize orchestrator
    orchestrator = AdaptiveInterviewOrchestrator()

    # Create a session
    session_id = "test_streak_session"
    resume_data = {
        "technical_skills": {"Python": 5, "JavaScript": 4},
        "projects": [{"project_name": "Test Project"}]
    }

    # Start interview
    logger.info("\n" + "=" * 80)
    logger.info("STEP 0: Start Interview")
    logger.info("=" * 80)
    result = orchestrator.start_interview(
        session_id=session_id,
        resume_data=resume_data,
        voice="af_bella"
    )

    # Get session to check initial streaks
    session = orchestrator.session_manager.get_session(session_id)
    if not session:
        logger.error("Session not created")
        return False

    logger.info(f"Initial streaks: correct={session.correct_streak}, wrong={session.wrong_streak}")
    assert session.correct_streak == 0, f"Initial correct_streak should be 0, got {session.correct_streak}"
    assert session.wrong_streak == 0, f"Initial wrong_streak should be 0, got {session.wrong_streak}"

    # Q1: Non-answer (incorrect)
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: Submit Q1 - Non-answer (incorrect)")
    logger.info("=" * 80)
    logger.info(f"Before Q1: correct={session.correct_streak}, wrong={session.wrong_streak}")

    result_q1 = orchestrator.process_answer(
        session_id=session_id,
        transcript="I don't know",  # Non-answer
        audio_duration=2.0,
        voice="af_bella"
    )

    logger.info(f"After Q1: correct={session.correct_streak}, wrong={session.wrong_streak}")
    logger.info(f"Expected Q1: correct=0, wrong=1")

    if session.correct_streak != 0 or session.wrong_streak != 1:
        logger.error(f"Q1 STREAK MISMATCH: got ({session.correct_streak}, {session.wrong_streak}), expected (0, 1)")
        return False

    # Q2: Non-answer (incorrect)
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: Submit Q2 - Non-answer (incorrect)")
    logger.info("=" * 80)
    logger.info(f"Before Q2: correct={session.correct_streak}, wrong={session.wrong_streak}")

    result_q2 = orchestrator.process_answer(
        session_id=session_id,
        transcript="Not sure",  # Non-answer
        audio_duration=2.0,
        voice="af_bella"
    )

    logger.info(f"After Q2: correct={session.correct_streak}, wrong={session.wrong_streak}")
    logger.info(f"Expected Q2: correct=0, wrong=2")

    if session.correct_streak != 0 or session.wrong_streak != 2:
        logger.error(f"Q2 STREAK MISMATCH: got ({session.correct_streak}, {session.wrong_streak}), expected (0, 2)")
        return False

    # Q3: Correct answer (must match the actual question asked)
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: Submit Q3 - Correct answer")
    logger.info("=" * 80)
    logger.info(f"Before Q3: correct={session.correct_streak}, wrong={session.wrong_streak}")

    # Get the actual question that was asked for Q3
    current_question = session.current_question
    logger.info(f"Q3 actual question: {current_question}")

    # Provide a relevant answer based on the question topic
    if "raise" in current_question.lower():
        answer = "The raise statement is used to trigger an exception explicitly in Python. It allows you to signal that an error condition has occurred and stop the normal flow of the program."
    elif "try-except" in current_question.lower():
        answer = "A try-except statement handles exceptions by catching them, while a finally block always executes regardless of whether an exception occurred, making it useful for cleanup operations."
    else:
        answer = "This is a correct and relevant answer to the question."

    result_q3 = orchestrator.process_answer(
        session_id=session_id,
        transcript=answer,
        audio_duration=5.0,
        voice="af_bella"
    )

    logger.info(f"After Q3: correct={session.correct_streak}, wrong={session.wrong_streak}")
    logger.info(f"Expected Q3: correct=1, wrong=0")

    if session.correct_streak != 1 or session.wrong_streak != 0:
        logger.error(f"Q3 STREAK MISMATCH: got ({session.correct_streak}, {session.wrong_streak}), expected (1, 0)")
        return False

    # Check research state history
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: Verify Research State History")
    logger.info("=" * 80)

    research_state = session.research_state_history
    logger.info(f"Number of research state records: {len(research_state)}")

    for i, state in enumerate(research_state):
        q_num = state['question_number']
        features = state['features']
        correct_streak = features['Correct Streak']
        wrong_streak = features['Wrong Streak']
        logger.info(f"Q{q_num} research state: correct_streak={correct_streak}, wrong_streak={wrong_streak}")

    # Expected: Q1 (0,1), Q2 (0,2), Q3 (1,0)
    if len(research_state) >= 3:
        q1_streaks = (research_state[0]['features']['Correct Streak'], research_state[0]['features']['Wrong Streak'])
        q2_streaks = (research_state[1]['features']['Correct Streak'], research_state[1]['features']['Wrong Streak'])
        q3_streaks = (research_state[2]['features']['Correct Streak'], research_state[2]['features']['Wrong Streak'])

        logger.info(f"Research state Q1: {q1_streaks}, expected (0, 1)")
        logger.info(f"Research state Q2: {q2_streaks}, expected (0, 2)")
        logger.info(f"Research state Q3: {q3_streaks}, expected (1, 0)")

        if q1_streaks != (0, 1) or q2_streaks != (0, 2) or q3_streaks != (1, 0):
            logger.error("RESEARCH STATE STREAK MISMATCH")
            return False

    logger.info("\n" + "=" * 80)
    logger.info("TEST PASSED: Streak updates are correct")
    logger.info("=" * 80)
    return True

if __name__ == "__main__":
    try:
        success = test_streak_updates()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Test failed with exception: {e}", exc_info=True)
        sys.exit(1)
