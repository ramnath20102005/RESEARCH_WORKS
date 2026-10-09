"""
Regression tests for no-answer detection in semantic evaluation.

Tests:
1. "I don't know." - Should trigger deterministic fallback
2. Completely irrelevant answer - Should be evaluated by LLM (not a no-answer)
3. Partial answer with two of four concepts explained - Should be evaluated by LLM
4. Correct answer with all four concepts explained - Should be evaluated by LLM
5. Incorrect answer - Should be evaluated by LLM
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app.llm import get_llm_client
from app.llm.semantic_evaluator import SemanticEvaluator
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_no_answer_regression():
    """Test no-answer detection regression cases."""
    
    logger.info("=" * 60)
    logger.info("NO-ANSWER DETECTION REGRESSION TESTS")
    logger.info("=" * 60)
    
    try:
        # Initialize semantic client
        logger.info("\n[TEST] Initializing semantic LLM client...")
        semantic_client = get_llm_client(task="semantic")
        logger.info(f"[TEST] Semantic client initialized")
        
        # Initialize semantic evaluator
        semantic_evaluator = SemanticEvaluator(semantic_client)
        
        # Test question and concepts
        question = "Explain the ACID properties in database transactions."
        topic = "Databases"
        current_difficulty = "Medium"
        required_concepts = ["Atomicity", "Consistency", "Isolation", "Durability"]
        
        # Regression test cases
        test_cases = [
            {
                "name": "Pure Non-Answer (short)",
                "answer": "I don't know.",
                "expected_behavior": "deterministic_fallback",
                "expected_scores": [5, 5, 5]
            },
            {
                "name": "Pure Non-Answer (medium)",
                "answer": "I do not know the answer to this question.",
                "expected_behavior": "deterministic_fallback",
                "expected_scores": [5, 5, 5]
            },
            {
                "name": "Irrelevant Answer",
                "answer": "The weather is nice today. I like going for walks in the park.",
                "expected_behavior": "llm_evaluation",
                "expected_scores": None  # Will be evaluated by LLM
            },
            {
                "name": "Partial Answer with 2 concepts",
                "answer": "ACID includes Atomicity and Consistency. Atomicity means transactions are all-or-nothing. Consistency means the database stays valid. I'm not sure about the other two properties.",
                "expected_behavior": "llm_evaluation",
                "expected_scores": None
            },
            {
                "name": "Correct Answer with all 4 concepts",
                "answer": "ACID stands for Atomicity, Consistency, Isolation, and Durability. Atomicity ensures that a transaction is all-or-nothing - either all operations succeed or none do. Consistency ensures the database moves from one valid state to another valid state. Isolation ensures concurrent transactions don't interfere with each other. Durability ensures that once a transaction commits, it persists even in case of system failure.",
                "expected_behavior": "llm_evaluation",
                "expected_scores": None
            },
            {
                "name": "Incorrect Answer",
                "answer": "ACID stands for Authentication, Cache, Indexing, and Distribution. These are security features for protecting databases from unauthorized access.",
                "expected_behavior": "llm_evaluation",
                "expected_scores": None
            }
        ]
        
        results = []
        
        for i, test_case in enumerate(test_cases, 1):
            logger.info(f"\n{'=' * 60}")
            logger.info(f"REGRESSION TEST {i}: {test_case['name']}")
            logger.info(f"Expected behavior: {test_case['expected_behavior']}")
            logger.info(f"{'=' * 60}")
            
            start_time = time.perf_counter()
            
            try:
                result = semantic_evaluator.evaluate_answer(
                    question=question,
                    topic=topic,
                    current_difficulty=current_difficulty,
                    answer=test_case['answer'],
                    required_concepts=required_concepts,
                    question_number=1,
                    correct_streak=0,
                    wrong_streak=0
                )
                
                elapsed = time.perf_counter() - start_time
                
                # Extract semantic features
                semantic = result['semantic']
                
                logger.info(f"\n[RESULT] Latency: {elapsed:.2f}s")
                logger.info(f"[RESULT] Correctness Score: {semantic['correctness_score']}")
                logger.info(f"[RESULT] Concept Coverage: {semantic['concept_coverage']}")
                logger.info(f"[RESULT] Reasoning Score: {semantic['reasoning_score']}")
                logger.info(f"[RESULT] Missing Concepts (count): {semantic['missing_concepts']}")
                logger.info(f"[RESULT] Missing Concepts (list): {semantic['missing_concept_list']}")
                logger.info(f"[RESULT] Is Correct: {semantic['is_correct']}")
                
                # Check if behavior matches expectation
                if test_case['expected_behavior'] == "deterministic_fallback":
                    behavior_match = (
                        semantic['correctness_score'] == 5 and
                        semantic['concept_coverage'] == 5 and
                        semantic['reasoning_score'] == 5
                    )
                    actual_behavior = "deterministic_fallback" if behavior_match else "llm_evaluation"
                else:
                    behavior_match = (
                        semantic['correctness_score'] != 5 or
                        semantic['concept_coverage'] != 5 or
                        semantic['reasoning_score'] != 5
                    )
                    actual_behavior = "llm_evaluation" if behavior_match else "deterministic_fallback"
                
                logger.info(f"[VALIDATION] Expected behavior: {test_case['expected_behavior']}")
                logger.info(f"[VALIDATION] Actual behavior: {actual_behavior}")
                logger.info(f"[VALIDATION] Behavior match: {behavior_match}")
                
                # Validate score ranges
                scores_valid = all(0 <= s <= 100 for s in [
                    semantic['correctness_score'],
                    semantic['concept_coverage'],
                    semantic['reasoning_score']
                ])
                
                # Validate count/list consistency
                count_match = semantic['missing_concepts'] == len(semantic['missing_concept_list'])
                
                results.append({
                    'name': test_case['name'],
                    'latency': elapsed,
                    'correctness': semantic['correctness_score'],
                    'coverage': semantic['concept_coverage'],
                    'reasoning': semantic['reasoning_score'],
                    'missing_count': semantic['missing_concepts'],
                    'missing_list_len': len(semantic['missing_concept_list']),
                    'count_match': count_match,
                    'expected_behavior': test_case['expected_behavior'],
                    'actual_behavior': actual_behavior,
                    'behavior_match': behavior_match,
                    'scores_valid': scores_valid,
                    'success': behavior_match and scores_valid and count_match
                })
                
            except Exception as e:
                elapsed = time.perf_counter() - start_time
                logger.error(f"[ERROR] Test case failed: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                
                results.append({
                    'name': test_case['name'],
                    'latency': elapsed,
                    'success': False,
                    'error': str(e)
                })
        
        # Summary
        logger.info(f"\n{'=' * 60}")
        logger.info("REGRESSION TEST SUMMARY")
        logger.info(f"{'=' * 60}")
        
        for result in results:
            if result['success']:
                logger.info(f"\n{result['name']}:")
                logger.info(f"  Expected behavior: {result['expected_behavior']}")
                logger.info(f"  Actual behavior: {result['actual_behavior']}")
                logger.info(f"  Behavior match: {result['behavior_match']}")
                logger.info(f"  Latency: {result['latency']:.2f}s")
                logger.info(f"  Scores: {result['correctness']}/{result['coverage']}/{result['reasoning']}")
                logger.info(f"  Missing: {result['missing_count']} (list: {result['missing_list_len']})")
                logger.info(f"  Count match: {result['count_match']}")
                logger.info(f"  Status: PASS")
            else:
                logger.info(f"\n{result['name']}:")
                logger.info(f"  Expected behavior: {result['expected_behavior']}")
                logger.info(f"  Actual behavior: {result.get('actual_behavior', 'N/A')}")
                logger.info(f"  Status: FAIL")
                logger.info(f"  Error: {result.get('error', 'Validation failed')}")
        
        pass_count = sum(1 for r in results if r['success'])
        logger.info(f"\nTotal: {pass_count}/{len(results)} tests passed")
        
        return results
        
    except Exception as e:
        logger.error(f"[TEST] Fatal error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None


if __name__ == "__main__":
    test_no_answer_regression()
