"""
Test script for complete end-to-end pipeline with new prompts.
Demonstrates the full flow: LLM features → 11-feature vector → TabPFN → policy → next question.
"""

import time
from app.llm.local_llm_client import LocalLLMClient
from app.llm.semantic_evaluator import SemanticEvaluator
from app.llm.question_generator import QuestionGenerator
from app.interview.tabpfn_inference import TabPFNInference
from app.interview.feature_builder import FeatureBuilder

print("=" * 70)
print("END-TO-END PIPELINE TEST WITH NEW PROMPTS")
print("=" * 70)

# Initialize components
print("\n[INITIALIZING COMPONENTS]")
llm_client = LocalLLMClient(
    model_name="Qwen/Qwen1.5-4B-Chat",
    quantization="int4"
)
semantic_evaluator = SemanticEvaluator(llm_client)
question_generator = QuestionGenerator(llm_client)
feature_builder = FeatureBuilder(random_seed=42)
tabpfn_inference = TabPFNInference()

# Simulate an interview scenario
print("\n" + "=" * 70)
print("STEP 1: SEMANTIC EVALUATION")
print("=" * 70)

question = "What is the difference between a Python list and tuple?"
topic = "Python"
current_difficulty = "Easy"
candidate_answer = "A list is mutable while a tuple is immutable. Lists can be modified after creation, but tuples cannot."

print(f"Question: {question}")
print(f"Topic: {topic}")
print(f"Difficulty: {current_difficulty}")
print(f"Candidate Answer: {candidate_answer}")

eval_start = time.perf_counter()
evaluation = semantic_evaluator.evaluate_answer(
    question=question,
    topic=topic,
    current_difficulty=current_difficulty,
    answer=candidate_answer,
    question_number=1,
    correct_streak=0,
    wrong_streak=0
)
eval_time = time.perf_counter() - eval_start

print(f"\nSemantic evaluation time: {eval_time*1000:.0f} ms")
print(f"\nSEMANTIC FEATURES:")
print(f"Correctness Score: {evaluation['semantic']['correctness_score']}")
print(f"Concept Coverage: {evaluation['semantic']['concept_coverage']}")
print(f"Reasoning Score: {evaluation['semantic']['reasoning_score']}")
print(f"Missing Concepts: {evaluation['semantic']['missing_concepts']}")
print(f"Question Difficulty: {evaluation['question_assessment']['question_difficulty']}")

print("\n" + "=" * 70)
print("STEP 2: FEATURE VECTOR CONSTRUCTION")
print("=" * 70)

feature_start = time.perf_counter()
feature_vector, feature_dict = feature_builder.build_feature_vector(
    llm_evaluation=evaluation,
    correct_streak=0,
    wrong_streak=0
)
feature_time = time.perf_counter() - feature_start

print(f"\nBEHAVIORAL FEATURES (random):")
print(f"Engagement Score: {feature_dict.get('engagement_score', 0.5):.3f}")
print(f"Confidence Score: {feature_dict.get('confidence_score', 0.5):.3f}")
print(f"Hesitation Score: {feature_dict.get('hesitation_score', 0.5):.3f}")
print(f"Eye Contact Score: {feature_dict.get('eye_contact_score', 0.5):.3f}")

print(f"\nCONTEXT FEATURES:")
print(f"Difficulty: {current_difficulty} (encoded: {feature_dict.get('difficulty_encoded', 1)})")
print(f"Correct Streak: 0")
print(f"Wrong Streak: 0")

print(f"\n11-FEATURE VECTOR:")
print(f"[{', '.join([str(x) for x in feature_vector])}]")

print("\n" + "=" * 70)
print("STEP 3: TABPFN POLICY PREDICTION")
print("=" * 70)

tabpfn_start = time.perf_counter()
policy_prediction = tabpfn_inference.predict_policy(
    feature_vector=feature_vector,
    return_probabilities=True
)
tabpfn_time = time.perf_counter() - tabpfn_start

print(f"TabPFN inference time: {tabpfn_time*1000:.0f} ms")

predicted_policy = policy_prediction['predicted_policy']
predicted_class = policy_prediction.get('predicted_class')
probabilities = policy_prediction.get('probabilities', {})

print(f"\nTABPFN OUTPUT:")
print(f"Predicted Class: {predicted_class}")
print(f"Predicted Policy: {predicted_policy}")
print(f"\nPolicy Probabilities:")
for policy, prob in probabilities.items():
    print(f"  {policy}: {prob:.4f}")

print("\n" + "=" * 70)
print("STEP 4: POLICY APPLICATION")
print("=" * 70)

# Apply policy to determine effective difficulty
difficulty_order = ["Easy", "Medium", "Hard"]
current_index = difficulty_order.index(current_difficulty)
effective_difficulty = current_difficulty

if predicted_policy == "Increase Difficulty":
    if current_index < len(difficulty_order) - 1:
        effective_difficulty = difficulty_order[current_index + 1]
elif predicted_policy == "Reduce Difficulty":
    if current_index > 0:
        effective_difficulty = difficulty_order[current_index - 1]

print(f"Current Difficulty: {current_difficulty}")
print(f"Predicted Policy: {predicted_policy}")
print(f"Effective Difficulty: {effective_difficulty}")

print("\n" + "=" * 70)
print("STEP 5: QUESTION GENERATION")
print("=" * 70)

question_start = time.perf_counter()
next_question_response = question_generator.generate_next_question(
    policy=predicted_policy,
    topic=topic,
    current_difficulty=effective_difficulty,
    previous_question=question,
    candidate_answer=candidate_answer,
    correctness_score=evaluation['semantic']['correctness_score'],
    concept_coverage=evaluation['semantic']['concept_coverage'],
    reasoning_score=evaluation['semantic']['reasoning_score'],
    missing_concepts=evaluation['semantic']['missing_concepts'],
    correct_streak=0,
    wrong_streak=0
)
question_time = time.perf_counter() - question_start

print(f"Question generation time: {question_time*1000:.0f} ms")
print(f"\nNEXT QUESTION:")
print(f"Question: {next_question_response.get('question')}")
print(f"Difficulty: {next_question_response.get('difficulty')}")
print(f"Topic: {next_question_response.get('topic')}")
print(f"Policy: {next_question_response.get('policy')}")

print("\n" + "=" * 70)
print("PIPELINE SUMMARY")
print("=" * 70)

print(f"\nCOMPLETE FLOW:")
print(f"1. LLM Semantic Evaluation → {evaluation['semantic']['correctness_score']}, {evaluation['semantic']['concept_coverage']}, {evaluation['semantic']['reasoning_score']}, {evaluation['semantic']['missing_concepts']}")
print(f"2. 11-Feature Vector → [{', '.join([str(x) for x in feature_vector])}]")
print(f"3. TabPFN Prediction → Class {predicted_class}, Policy: {predicted_policy}")
print(f"4. Policy Application → {current_difficulty} → {effective_difficulty}")
print(f"5. Question Generation → Difficulty: {next_question_response.get('difficulty')}, Policy: {next_question_response.get('policy')}")

print(f"\nPOLICY VERIFICATION:")
policy_match = next_question_response.get('policy') == predicted_policy
print(f"TabPFN Policy: {predicted_policy}")
print(f"Question Policy: {next_question_response.get('policy')}")
print(f"Policy Match: {policy_match}")

print(f"\nDIFFICULTY VERIFICATION:")
difficulty_match = next_question_response.get('difficulty') == effective_difficulty
print(f"Effective Difficulty: {effective_difficulty}")
print(f"Question Difficulty: {next_question_response.get('difficulty')}")
print(f"Difficulty Match: {difficulty_match}")

total_time = eval_time + feature_time + tabpfn_time + question_time
print(f"\nTOTAL PIPELINE TIME: {total_time*1000:.0f} ms")
print(f"  - Semantic Evaluation: {eval_time*1000:.0f} ms")
print(f"  - Feature Building: {feature_time*1000:.0f} ms")
print(f"  - TabPFN Inference: {tabpfn_time*1000:.0f} ms")
print(f"  - Question Generation: {question_time*1000:.0f} ms")

print("\n" + "=" * 70)
print("END-TO-END PIPELINE TEST COMPLETE")
print("=" * 70)
