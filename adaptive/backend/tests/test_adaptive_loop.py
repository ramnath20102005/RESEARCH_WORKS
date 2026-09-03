"""
Test script for the adaptive interview loop.

Tests the complete flow:
1. Session creation
2. First question generation
3. Answer processing
4. LLM evaluation
5. Feature vector construction
6. TabPFN prediction
7. Next question generation
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.llm.nvidia_client import NVIDIAClient
from app.interview.adaptive_orchestrator import AdaptiveInterviewOrchestrator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_adaptive_loop():
    """Test the complete adaptive interview loop."""
    
    logger.info("=" * 60)
    logger.info("ADAPTIVE INTERVIEW LOOP TEST")
    logger.info("=" * 60)
    
    try:
        # Initialize LLM client
        logger.info("\n[TEST] Initializing NVIDIA client...")
        llm_client = NVIDIAClient()
        
        # Initialize orchestrator
        logger.info("[TEST] Initializing adaptive orchestrator...")
        orchestrator = AdaptiveInterviewOrchestrator(llm_client=llm_client)
        
        # Test resume data
        resume_data = {
            "name": "Test Candidate",
            "technical_skills": {
                "Python": "Advanced",
                "Machine Learning": "Intermediate",
                "SQL": "Advanced"
            },
            "projects": [
                {
                    "name": "ML Project",
                    "description": "Built a classification model"
                }
            ],
            "area_of_interest": ["Machine Learning", "Data Science"]
        }
        
        # Test 1: Start interview
        logger.info("\n[TEST 1] Starting interview...")
        session_id = "test_session_001"
        result = orchestrator.start_interview(session_id, resume_data)
        
        logger.info(f"[TEST 1] ✓ Interview started")
        logger.info(f"  Question: {result['question'][:50]}...")
        logger.info(f"  Topic: {result['topic']}")
        logger.info(f"  Difficulty: {result['difficulty']}")
        logger.info(f"  TTS Audio: {len(result['tts_audio'])} bytes")
        
        # Test 2: Process answer (high quality)
        logger.info("\n[TEST 2] Processing high-quality answer...")
        transcript = "A primary key is a column or set of columns that uniquely identifies each row in a database table. It ensures data integrity and allows for efficient data retrieval. Primary keys cannot contain NULL values and must be unique for each record."
        
        result = orchestrator.process_answer(session_id, transcript)
        
        logger.info(f"[TEST 2] ✓ Answer processed")
        logger.info(f"  Correctness: {result['evaluation']['semantic']['correctness_score']}")
        logger.info(f"  Coverage: {result['evaluation']['semantic']['concept_coverage']}")
        logger.info(f"  Reasoning: {result['evaluation']['semantic']['reasoning_score']}")
        logger.info(f"  Missing Concepts: {result['evaluation']['semantic']['missing_concepts']}")
        logger.info(f"  Question Difficulty: {result['evaluation']['question_assessment']['question_difficulty']}")
        logger.info(f"  Predicted Policy: {result['policy']}")
        logger.info(f"  Next Question: {result['next_question'][:50]}...")
        logger.info(f"  Next Difficulty: {result['next_difficulty']}")
        
        # Test 3: Process answer (low quality)
        logger.info("\n[TEST 3] Processing low-quality answer...")
        transcript = "I don't know much about databases but I think primary keys are important for something."
        
        result = orchestrator.process_answer(session_id, transcript)
        
        logger.info(f"[TEST 3] ✓ Answer processed")
        logger.info(f"  Correctness: {result['evaluation']['semantic']['correctness_score']}")
        logger.info(f"  Coverage: {result['evaluation']['semantic']['concept_coverage']}")
        logger.info(f"  Reasoning: {result['evaluation']['semantic']['reasoning_score']}")
        logger.info(f"  Missing Concepts: {result['evaluation']['semantic']['missing_concepts']}")
        logger.info(f"  Predicted Policy: {result['policy']}")
        logger.info(f"  Next Difficulty: {result['next_difficulty']}")
        
        # Test 4: Get session state
        logger.info("\n[TEST 4] Getting session state...")
        state = orchestrator.get_session_state(session_id)
        
        logger.info(f"[TEST 4] ✓ Session state retrieved")
        logger.info(f"  Question Number: {state['context']['question_number']}")
        logger.info(f"  Current Difficulty: {state['context']['current_difficulty']}")
        logger.info(f"  Correct Streak: {state['context']['correct_streak']}")
        logger.info(f"  Wrong Streak: {state['context']['wrong_streak']}")
        
        # Test 5: End session
        logger.info("\n[TEST 5] Ending session...")
        summary = orchestrator.end_session(session_id)
        
        logger.info(f"[TEST 5] ✓ Session ended")
        logger.info(f"  Total Questions: {summary['session_summary']['total_questions']}")
        logger.info(f"  Average Correctness: {summary['session_summary']['avg_correctness']}")
        logger.info(f"  Accuracy: {summary['session_summary']['accuracy']}")
        
        logger.info("\n" + "=" * 60)
        logger.info("ALL TESTS PASSED ✓")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"\n[TEST] FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_adaptive_loop()
    sys.exit(0 if success else 1)
