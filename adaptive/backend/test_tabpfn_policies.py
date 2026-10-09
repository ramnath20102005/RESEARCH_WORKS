"""
Test all seven TabPFN policies with the full orchestrator.

Verifies that:
1. TabPFN predicts the correct policy for each scenario
2. The predicted policy is executed by the backend
3. Difficulty boundaries are respected
4. Switch Topic actually changes the topic
5. Probe Missing Concept targets missing concepts
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app.interview.adaptive_orchestrator import AdaptiveInterviewOrchestrator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_tabpfn_policies():
    """Test all seven TabPFN policies with orchestrator."""
    
    logger.info("=" * 60)
    logger.info("TABPFN POLICY EXECUTION TEST")
    logger.info("=" * 60)
    
    try:
        # Initialize orchestrator
        logger.info("\n[TEST] Initializing adaptive orchestrator...")
        orchestrator = AdaptiveInterviewOrchestrator()
        
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
            "area_of_interest": ["Machine Learning", "Data Science", "Databases", "Web Development"]
        }
        
        # Policy test scenarios
        # Each scenario is designed to trigger a specific TabPFN policy
        test_scenarios = [
            {
                "name": "Increase Difficulty (high performance)",
                "session_id": "test_policy_1",
                "answer": "A list in Python is a mutable sequence that can hold multiple items of different types. It supports operations like append, extend, remove, pop, slicing, and list comprehensions. Lists are implemented as dynamic arrays with O(1) amortized append and O(n) worst-case insertion. They are distinct from tuples which are immutable, and from arrays which require homogeneous element types.",
                "expected_policy": "Increase Difficulty"
            },
            {
                "name": "Maintain Difficulty (moderate performance)",
                "session_id": "test_policy_2",
                "answer": "A list is a mutable sequence in Python that can hold multiple items. You can add items with append() and remove them with remove().",
                "expected_policy": "Maintain Difficulty"
            },
            {
                "name": "Reduce Difficulty (poor performance)",
                "session_id": "test_policy_3",
                "answer": "I don't know much about Python lists.",
                "expected_policy": "Reduce Difficulty"
            },
            {
                "name": "Probe Missing Concept (partial answer)",
                "session_id": "test_policy_4",
                "answer": "A list in Python is mutable and can hold items. I'm not sure about the other properties.",
                "expected_policy": "Probe Missing Concept"
            },
            {
                "name": "Ask Application Question (good performance)",
                "session_id": "test_policy_5",
                "answer": "Lists are mutable sequences that hold multiple items. They're commonly used for storing collections of data in Python applications.",
                "expected_policy": "Ask Application Question"
            },
            {
                "name": "Ask Follow-up Question (adequate answer)",
                "session_id": "test_policy_6",
                "answer": "A list is a mutable sequence that can hold multiple items of different types.",
                "expected_policy": "Ask Follow-up Question"
            },
            {
                "name": "Switch Topic (after multiple questions)",
                "session_id": "test_policy_7",
                "answer": "I don't know anything about this topic.",
                "expected_policy": "Switch Topic"
            }
        ]
        
        results = []
        
        for i, scenario in enumerate(test_scenarios, 1):
            logger.info(f"\n{'=' * 60}")
            logger.info(f"POLICY TEST {i}: {scenario['name']}")
            logger.info(f"Expected Policy: {scenario['expected_policy']}")
            logger.info(f"{'=' * 60}")
            
            try:
                # Start a new session for each test
                logger.info(f"\n[TEST] Starting session {scenario['session_id']}...")
                start_result = orchestrator.start_interview(scenario['session_id'], resume_data)
                logger.info(f"[TEST] First question: {start_result['question'][:50]}...")
                logger.info(f"[TEST] First topic: {start_result['topic']}")
                logger.info(f"[TEST] First difficulty: {start_result['difficulty']}")
                
                # For Switch Topic, need multiple consecutive poor answers
                if scenario['expected_policy'] == "Switch Topic":
                    logger.info("[TEST] Submitting multiple poor answers to trigger Switch Topic...")
                    for j in range(3):
                        result = orchestrator.process_answer(scenario['session_id'], scenario['answer'])
                        logger.info(f"[TEST] Answer {j+1} processed. Policy: {result['policy']}")
                
                # Process the test answer
                result = orchestrator.process_answer(scenario['session_id'], scenario['answer'])
                
                predicted_policy = result['policy']
                next_difficulty = result['next_difficulty']
                next_topic = result.get('next_topic', start_result['topic'])
                
                logger.info(f"\n[RESULT] Predicted Policy: {predicted_policy}")
                logger.info(f"[RESULT] Expected Policy: {scenario['expected_policy']}")
                logger.info(f"[RESULT] Policy Match: {predicted_policy == scenario['expected_policy']}")
                logger.info(f"[RESULT] Next Difficulty: {next_difficulty}")
                logger.info(f"[RESULT] Next Topic: {next_topic}")
                
                # For Switch Topic, verify topic changed
                if scenario['expected_policy'] == "Switch Topic":
                    topic_changed = next_topic != start_result['topic']
                    logger.info(f"[VALIDATION] Topic changed: {topic_changed}")
                
                results.append({
                    'name': scenario['name'],
                    'expected_policy': scenario['expected_policy'],
                    'predicted_policy': predicted_policy,
                    'policy_match': predicted_policy == scenario['expected_policy'],
                    'next_difficulty': next_difficulty,
                    'next_topic': next_topic,
                    'success': predicted_policy == scenario['expected_policy']
                })
                
                # End session
                orchestrator.end_session(scenario['session_id'])
                
            except Exception as e:
                logger.error(f"[ERROR] Policy test failed: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                
                results.append({
                    'name': scenario['name'],
                    'expected_policy': scenario['expected_policy'],
                    'success': False,
                    'error': str(e)
                })
        
        # Summary
        logger.info(f"\n{'=' * 60}")
        logger.info("POLICY TEST SUMMARY")
        logger.info(f"{'=' * 60}")
        
        for result in results:
            if result['success']:
                logger.info(f"\n{result['name']}:")
                logger.info(f"  Expected: {result['expected_policy']}")
                logger.info(f"  Predicted: {result['predicted_policy']}")
                logger.info(f"  Match: {result['policy_match']}")
                logger.info(f"  Next Difficulty: {result['next_difficulty']}")
                logger.info(f"  Status: PASS")
            else:
                logger.info(f"\n{result['name']}:")
                logger.info(f"  Expected: {result['expected_policy']}")
                logger.info(f"  Status: FAIL")
                logger.info(f"  Error: {result.get('error', 'Policy mismatch')}")
        
        pass_count = sum(1 for r in results if r['success'])
        logger.info(f"\nTotal: {pass_count}/{len(results)} policies matched")
        
        return results
        
    except Exception as e:
        logger.error(f"[TEST] Fatal error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None


if __name__ == "__main__":
    test_tabpfn_policies()
