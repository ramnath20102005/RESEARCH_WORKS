"""
Test script to verify research state tracking for the adaptive interview system.

This test verifies that:
1. Every question-answer cycle captures all 11 features
2. The exact 11-feature vector used by TabPFN is stored
3. The selected policy is recorded
4. Behavioral values match the backend-generated values
5. Multiple question-answer cycles maintain consistent state tracking
"""

import sys
from pathlib import Path

# Add app directory to path
app_dir = Path(__file__).parent.parent
sys.path.insert(0, str(app_dir))

from app.interview.session_manager import SessionManager, InterviewSession
from app.interview.feature_builder import FeatureBuilder
from app.interview.tabpfn_inference import TabPFNInference
from unittest.mock import Mock, MagicMock
import json


def test_research_state_tracking():
    """Test that research state tracking captures all 11 features correctly."""
    
    print("=" * 70)
    print("TEST: Research State Tracking")
    print("=" * 70)
    
    # Create session manager
    session_manager = SessionManager()
    session = session_manager.create_session("test_session_001")
    
    # Mock resume data
    resume_data = {
        "skills": ["Python", "FastAPI", "MongoDB"],
        "projects": [{"name": "Test Project", "tech_stack": ["Python"]}]
    }
    
    # Add first question
    session.add_question(
        question="What is the difference between a list and tuple in Python?",
        topic="Python",
        difficulty="Easy",
        source="Skill",
        required_concepts=["mutability", "syntax", "use cases"]
    )
    
    # Simulate candidate answer
    session.add_answer("A list is mutable while a tuple is immutable.", audio_duration=5.2)
    
    # Simulate semantic evaluation
    evaluation = {
        'semantic': {
            'correctness_score': 85,
            'concept_coverage': 80,
            'reasoning_score': 82,
            'missing_concepts': 2,
            'missing_concept_list': ["use cases", "memory differences"],
            'is_correct': True
        },
        'question_assessment': {
            'question_difficulty': 'Easy'
        }
    }
    session.add_evaluation(evaluation)
    
    # Build feature vector (simulating the actual process)
    feature_builder = FeatureBuilder(random_seed=42)
    feature_vector, feature_dict = feature_builder.build_feature_vector(
        llm_evaluation=evaluation,
        correct_streak=session.correct_streak,
        wrong_streak=session.wrong_streak
    )
    
    # Simulate TabPFN policy selection
    selected_policy = "Increase Difficulty"
    session.add_policy(selected_policy)
    
    # Record research state
    research_features = {
        'correctness_score': evaluation['semantic']['correctness_score'],
        'concept_coverage': evaluation['semantic']['concept_coverage'],
        'reasoning_score': evaluation['semantic']['reasoning_score'],
        'missing_concepts': evaluation['semantic']['missing_concepts'],
        'engagement_score': feature_dict['Engagement Score'],
        'confidence_score': feature_dict['Confidence Score'],
        'hesitation_score': feature_dict['Hesitation Score'],
        'eye_contact_score': feature_dict['Eye Contact Score'],
        'difficulty': feature_dict['Difficulty'],
        'correct_streak': feature_dict['Correct Streak'],
        'wrong_streak': feature_dict['Wrong Streak']
    }
    
    session.add_research_state(
        features=research_features,
        tabpfn_input_vector=feature_vector,
        selected_policy=selected_policy
    )
    
    # Verify research state
    research_history = session.get_research_state_history()
    
    print(f"\n✓ Research state records: {len(research_history)}")
    assert len(research_history) == 1, "Should have 1 research state record"
    
    # Verify all 11 features are present
    state = research_history[0]
    features = state['features']
    
    print(f"\n✓ Features captured: {len(features)}")
    assert len(features) == 11, f"Should have 11 features, got {len(features)}"
    
    # Verify feature names
    expected_features = [
        'correctness_score', 'concept_coverage', 'reasoning_score', 'missing_concepts',
        'engagement_score', 'confidence_score', 'hesitation_score', 'eye_contact_score',
        'difficulty', 'correct_streak', 'wrong_streak'
    ]
    
    for feature in expected_features:
        assert feature in features, f"Missing feature: {feature}"
        print(f"  ✓ {feature}: {features[feature]}")
    
    # Verify TabPFN input vector
    tabpfn_vector = state['tabpfn_input_vector']
    print(f"\n✓ TabPFN input vector length: {len(tabpfn_vector)}")
    assert len(tabpfn_vector) == 11, f"TabPFN vector should have 11 elements, got {len(tabpfn_vector)}"
    
    # Verify the vector matches the features
    assert tabpfn_vector[0] == features['correctness_score'], "Vector[0] should match correctness_score"
    assert tabpfn_vector[1] == features['concept_coverage'], "Vector[1] should match concept_coverage"
    assert tabpfn_vector[2] == features['reasoning_score'], "Vector[2] should match reasoning_score"
    assert tabpfn_vector[3] == features['missing_concepts'], "Vector[3] should match missing_concepts"
    assert tabpfn_vector[4] == features['engagement_score'], "Vector[4] should match engagement_score"
    assert tabpfn_vector[5] == features['confidence_score'], "Vector[5] should match confidence_score"
    assert tabpfn_vector[6] == features['hesitation_score'], "Vector[6] should match hesitation_score"
    assert tabpfn_vector[7] == features['eye_contact_score'], "Vector[7] should match eye_contact_score"
    assert tabpfn_vector[8] == features['difficulty'], "Vector[8] should match difficulty"
    assert tabpfn_vector[9] == features['correct_streak'], "Vector[9] should match correct_streak"
    assert tabpfn_vector[10] == features['wrong_streak'], "Vector[10] should match wrong_streak"
    
    print("  ✓ TabPFN vector matches feature values")
    
    # Verify selected policy
    assert state['selected_policy'] == selected_policy, f"Selected policy mismatch"
    print(f"  ✓ Selected policy: {selected_policy}")
    
    # Simulate second question-answer cycle
    session.add_question(
        question="Can you explain mutability in more detail?",
        topic="Python",
        difficulty="Medium",
        source="Follow-up",
        required_concepts=["mutability", "examples"]
    )
    
    session.add_answer("Mutability means you can change the object after creation.", audio_duration=4.1)
    
    evaluation2 = {
        'semantic': {
            'correctness_score': 92,
            'concept_coverage': 88,
            'reasoning_score': 90,
            'missing_concepts': 1,
            'missing_concept_list': ["examples"],
            'is_correct': True
        },
        'question_assessment': {
            'question_difficulty': 'Medium'
        }
    }
    session.add_evaluation(evaluation2)
    
    feature_vector2, feature_dict2 = feature_builder.build_feature_vector(
        llm_evaluation=evaluation2,
        correct_streak=session.correct_streak,
        wrong_streak=session.wrong_streak
    )
    
    selected_policy2 = "Maintain Difficulty"
    session.add_policy(selected_policy2)
    
    research_features2 = {
        'correctness_score': evaluation2['semantic']['correctness_score'],
        'concept_coverage': evaluation2['semantic']['concept_coverage'],
        'reasoning_score': evaluation2['semantic']['reasoning_score'],
        'missing_concepts': evaluation2['semantic']['missing_concepts'],
        'engagement_score': feature_dict2['Engagement Score'],
        'confidence_score': feature_dict2['Confidence Score'],
        'hesitation_score': feature_dict2['Hesitation Score'],
        'eye_contact_score': feature_dict2['Eye Contact Score'],
        'difficulty': feature_dict2['Difficulty'],
        'correct_streak': feature_dict2['Correct Streak'],
        'wrong_streak': feature_dict2['Wrong Streak']
    }
    
    session.add_research_state(
        features=research_features2,
        tabpfn_input_vector=feature_vector2,
        selected_policy=selected_policy2
    )
    
    # Verify multiple states
    research_history = session.get_research_state_history()
    print(f"\n✓ Research state records after 2 cycles: {len(research_history)}")
    assert len(research_history) == 2, "Should have 2 research state records"
    
    # Verify each state has complete data
    for i, state in enumerate(research_history, 1):
        print(f"\n--- Question {i} ---")
        print(f"  Question: {state['question'][:50]}...")
        print(f"  Features: {len(state['features'])}")
        print(f"  TabPFN vector: {len(state['tabpfn_input_vector'])}")
        print(f"  Selected policy: {state['selected_policy']}")
        
        assert len(state['features']) == 11
        assert len(state['tabpfn_input_vector']) == 11
        assert state['selected_policy'] in [
            "Increase Difficulty", "Maintain Difficulty", "Reduce Difficulty",
            "Probe Missing Concept", "Ask Application Question", 
            "Ask Follow-up Question", "Switch Topic"
        ]
    
    print("\n" + "=" * 70)
    print("✓ ALL TESTS PASSED")
    print("=" * 70)
    print("\nResearch state tracking successfully:")
    print("  1. Captures all 11 features per question-answer cycle")
    print("  2. Stores exact TabPFN input vector")
    print("  3. Records selected policy")
    print("  4. Maintains consistent state across multiple cycles")
    print("  5. Behavioral values match backend-generated values")
    
    # Print sample research table
    print("\n" + "=" * 70)
    print("SAMPLE RESEARCH TABLE")
    print("=" * 70)
    print("Q | Corr | Cov | Reas | Miss | Eng | Conf | Hes | Eye | Diff | Str | WStr | Policy")
    print("-" * 90)
    
    for i, state in enumerate(research_history, 1):
        f = state['features']
        print(f"Q{i} | {f['correctness_score']:3.0f} | {f['concept_coverage']:3.0f} | "
              f"{f['reasoning_score']:3.0f} | {f['missing_concepts']:3.0f} | "
              f"{f['engagement_score']:3.2f} | {f['confidence_score']:3.2f} | "
              f"{f['hesitation_score']:3.2f} | {f['eye_contact_score']:3.2f} | "
              f"{f['difficulty']:3.0f} | {f['correct_streak']:3.0f} | "
              f"{f['wrong_streak']:3.0f} | {state['selected_policy']}")
    
    print("=" * 70)
    
    return True


if __name__ == "__main__":
    try:
        test_research_state_tracking()
        print("\n✓ Research state tracking implementation verified successfully")
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
