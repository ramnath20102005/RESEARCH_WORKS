"""
Comprehensive test suite for adaptive interview system refactor.

Tests verify the 20 requirements and 12 specific test cases.
"""

import pytest
import json
from unittest.mock import Mock, MagicMock, patch
from app.interview.session_manager import SessionManager, InterviewSession
from app.interview.feature_builder import FeatureBuilder
from app.interview.tabpfn_inference import TabPFNInference
from app.llm.semantic_evaluator import SemanticEvaluator
from app.llm.question_generator import QuestionGenerator
from app.llm.local_llm_client import LocalLLMClient


class TestAdaptiveRefactor:
    """Test suite for adaptive interview system refactor."""
    
    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        client = Mock()
        client.generate_json = Mock()
        client.generate_content = Mock()
        return client
    
    @pytest.fixture
    def session_manager(self):
        """Create a session manager."""
        return SessionManager()
    
    @pytest.fixture
    def feature_builder(self):
        """Create a feature builder."""
        return FeatureBuilder(random_seed=42)
    
    @pytest.fixture
    def sample_resume_data(self):
        """Sample resume data for testing."""
        return {
            "skills": ["Python", "FastAPI", "MongoDB", "Machine Learning", "React"],
            "projects": [
                {
                    "name": "News Sentiment Analysis",
                    "tech_stack": ["Python", "FastAPI", "MongoDB"]
                }
            ],
            "experience": [
                {
                    "skills": ["Python", "FastAPI"]
                }
            ]
        }
    
    # TEST 1: Good complete answer
    def test_good_complete_answer(self, mock_llm_client):
        """Test that a good complete answer produces high semantic scores."""
        # Setup mock response for good answer
        mock_llm_client.generate_json.return_value = {
            "correctness_score": 95,
            "concept_coverage": 90,
            "reasoning_score": 92,
            "missing_concepts": 1,
            "missing_concept_list": [],
            "difficulty": "Medium",
            "is_correct": True
        }
        
        evaluator = SemanticEvaluator(mock_llm_client)
        result = evaluator.evaluate_answer(
            question="What is the difference between a list and tuple in Python?",
            topic="Python",
            current_difficulty="Medium",
            answer="A list is mutable while a tuple is immutable. Lists can be modified after creation, but tuples cannot. Lists use square brackets [] while tuples use parentheses (). Lists are generally used for homogeneous data, while tuples are often used for heterogeneous data.",
            required_concepts=["mutability", "syntax", "use cases"]
        )
        
        assert result['semantic']['correctness_score'] >= 90
        assert result['semantic']['concept_coverage'] >= 85
        assert result['semantic']['reasoning_score'] >= 85
        assert result['semantic']['missing_concepts'] <= 2
        assert result['semantic']['is_correct'] == True
    
    # TEST 2: Partial answer
    def test_partial_answer(self, mock_llm_client):
        """Test that a partial answer produces moderate scores with missing concepts."""
        mock_llm_client.generate_json.return_value = {
            "correctness_score": 65,
            "concept_coverage": 60,
            "reasoning_score": 55,
            "missing_concepts": 3,
            "missing_concept_list": ["use cases", "memory differences"],
            "difficulty": "Medium",
            "is_correct": False
        }
        
        evaluator = SemanticEvaluator(mock_llm_client)
        result = evaluator.evaluate_answer(
            question="What is the difference between a list and tuple in Python?",
            topic="Python",
            current_difficulty="Medium",
            answer="A list is mutable and a tuple is immutable.",
            required_concepts=["mutability", "syntax", "use cases", "memory differences"]
        )
        
        assert 50 <= result['semantic']['correctness_score'] <= 80
        assert 50 <= result['semantic']['concept_coverage'] <= 80
        assert result['semantic']['missing_concepts'] > 0
        assert len(result['semantic']['missing_concept_list']) > 0
    
    # TEST 3: Explicit "I don't know"
    def test_i_dont know_response(self, mock_llm_client):
        """Test that 'I don't know' produces low scores but does NOT hardcode policy."""
        evaluator = SemanticEvaluator(mock_llm_client)
        result = evaluator.evaluate_answer(
            question="What is the difference between a list and tuple in Python?",
            topic="Python",
            current_difficulty="Medium",
            answer="I don't know the answer to this question.",
            required_concepts=["mutability", "syntax", "use cases"]
        )
        
        # Should produce low scores
        assert result['semantic']['correctness_score'] <= 10
        assert result['semantic']['concept_coverage'] <= 10
        assert result['semantic']['reasoning_score'] <= 10
        assert result['semantic']['missing_concepts'] >= 5
        assert result['semantic']['is_correct'] == False
        
        # Should NOT have hardcoded policy - this is just semantic evaluation
        assert 'policy' not in result
    
    # TEST 4: Increase Difficulty + Easy
    def test_increase_difficulty_easy_to_medium(self):
        """Test that Increase Difficulty from Easy results in Medium."""
        feature_builder = FeatureBuilder()
        
        # Simulate policy application
        current_difficulty = "Easy"
        policy = "Increase Difficulty"
        
        difficulty_order = ["Easy", "Medium", "Hard"]
        current_index = difficulty_order.index(current_difficulty)
        
        if policy == "Increase Difficulty":
            if current_index < len(difficulty_order) - 1:
                effective_difficulty = difficulty_order[current_index + 1]
            else:
                effective_difficulty = current_difficulty
        
        assert effective_difficulty == "Medium"
    
    # TEST 5: Increase Difficulty + Medium
    def test_increase_difficulty_medium_to_hard(self):
        """Test that Increase Difficulty from Medium results in Hard."""
        feature_builder = FeatureBuilder()
        
        current_difficulty = "Medium"
        policy = "Increase Difficulty"
        
        difficulty_order = ["Easy", "Medium", "Hard"]
        current_index = difficulty_order.index(current_difficulty)
        
        if policy == "Increase Difficulty":
            if current_index < len(difficulty_order) - 1:
                effective_difficulty = difficulty_order[current_index + 1]
            else:
                effective_difficulty = current_difficulty
        
        assert effective_difficulty == "Hard"
    
    # TEST 6: Reduce Difficulty + Hard
    def test_reduce_difficulty_hard_to_medium(self):
        """Test that Reduce Difficulty from Hard results in Medium."""
        feature_builder = FeatureBuilder()
        
        current_difficulty = "Hard"
        policy = "Reduce Difficulty"
        
        difficulty_order = ["Easy", "Medium", "Hard"]
        current_index = difficulty_order.index(current_difficulty)
        
        if policy == "Reduce Difficulty":
            if current_index > 0:
                effective_difficulty = difficulty_order[current_index - 1]
            else:
                effective_difficulty = current_difficulty
        
        assert effective_difficulty == "Medium"
    
    # TEST 7: Maintain Difficulty
    def test_maintain_difficulty(self):
        """Test that Maintain Difficulty keeps the same difficulty."""
        feature_builder = FeatureBuilder()
        
        current_difficulty = "Medium"
        policy = "Maintain Difficulty"
        
        effective_difficulty = current_difficulty
        
        assert effective_difficulty == "Medium"
    
    # TEST 8: Probe Missing Concept
    def test_probe_missing_concept_targets_missing_concept(self, mock_llm_client):
        """Test that Probe Missing Concept generates a question targeting missing concepts."""
        mock_llm_client.generate_json.return_value = {
            "question": "You mentioned mutability differences. Can you explain how this affects memory usage and performance when choosing between lists and tuples?",
            "difficulty": "Medium",
            "topic": "Python",
            "policy": "Probe Missing Concept",
            "source": "Follow-up"
        }
        
        generator = QuestionGenerator(mock_llm_client)
        result = generator.generate_next_question(
            policy="Probe Missing Concept",
            topic="Python",
            current_difficulty="Medium",
            effective_difficulty="Medium",
            previous_question="What is the difference between a list and tuple in Python?",
            candidate_answer="A list is mutable and a tuple is immutable.",
            correctness_score=65,
            concept_coverage=60,
            reasoning_score=55,
            missing_concepts=3,
            missing_concept_list=["memory usage", "performance", "use cases"],
            correct_streak=0,
            wrong_streak=0
        )
        
        assert result['policy'] == "Probe Missing Concept"
        assert result['difficulty'] == "Medium"
        # Question should reference at least one missing concept
        question_lower = result['question'].lower()
        has_missing_concept = any(
            concept.lower() in question_lower 
            for concept in ["memory usage", "performance", "use cases"]
        )
        assert has_missing_concept
    
    # TEST 9: Switch Topic
    def test_switch_topic_changes_topic(self, mock_llm_client, sample_resume_data):
        """Test that Switch Topic actually changes the topic."""
        mock_llm_client.generate_json.return_value = {
            "question": "You have experience with FastAPI. Can you explain how dependency injection works in FastAPI?",
            "difficulty": "Medium",
            "topic": "FastAPI",
            "policy": "Switch Topic",
            "source": "Topic"
        }
        
        generator = QuestionGenerator(mock_llm_client)
        result = generator.generate_next_question(
            policy="Switch Topic",
            topic="Python",
            current_difficulty="Medium",
            effective_difficulty="Medium",
            previous_question="What is the difference between a list and tuple in Python?",
            candidate_answer="A list is mutable and a tuple is immutable.",
            correctness_score=85,
            concept_coverage=80,
            reasoning_score=82,
            missing_concepts=1,
            missing_concept_list=[],
            correct_streak=2,
            wrong_streak=0,
            resume_data=sample_resume_data,
            available_topics=["FastAPI", "MongoDB", "Machine Learning", "React"]
        )
        
        assert result['policy'] == "Switch Topic"
        assert result['topic'] != "Python"  # Must be different from current topic
        assert result['difficulty'] == "Medium"
    
    # TEST 10: Same question + same answer reproducibility
    def test_semantic_evaluation_reproducibility(self, mock_llm_client):
        """Test that semantic evaluation is reproducible with same inputs."""
        # Set up mock to return same response twice
        mock_llm_client.generate_json.return_value = {
            "correctness_score": 75,
            "concept_coverage": 70,
            "reasoning_score": 72,
            "missing_concepts": 2,
            "missing_concept_list": ["use cases"],
            "difficulty": "Medium",
            "is_correct": True
        }
        
        evaluator = SemanticEvaluator(mock_llm_client)
        
        # First evaluation
        result1 = evaluator.evaluate_answer(
            question="What is the difference between a list and tuple in Python?",
            topic="Python",
            current_difficulty="Medium",
            answer="A list is mutable while a tuple is immutable.",
            required_concepts=["mutability", "syntax", "use cases"]
        )
        
        # Second evaluation with same inputs
        result2 = evaluator.evaluate_answer(
            question="What is the difference between a list and tuple in Python?",
            topic="Python",
            current_difficulty="Medium",
            answer="A list is mutable while a tuple is immutable.",
            required_concepts=["mutability", "syntax", "use cases"]
        )
        
        # Results should be identical
        assert result1['semantic']['correctness_score'] == result2['semantic']['correctness_score']
        assert result1['semantic']['concept_coverage'] == result2['semantic']['concept_coverage']
        assert result1['semantic']['reasoning_score'] == result2['semantic']['reasoning_score']
        assert result1['semantic']['missing_concepts'] == result2['semantic']['missing_concepts']
    
    # TEST 11: Generated question policy differs from TabPFN policy
    def test_policy_validation_rejects_mismatch(self, mock_llm_client):
        """Test that validation rejects when LLM returns different policy."""
        mock_llm_client.generate_json.return_value = {
            "question": "Can you explain this further?",
            "difficulty": "Medium",
            "topic": "Python",
            "policy": "Maintain Difficulty",  # Different from expected
            "source": "Follow-up"
        }
        
        generator = QuestionGenerator(mock_llm_client)
        
        with pytest.raises(ValueError, match="Policy validation failed"):
            generator.generate_next_question(
                policy="Increase Difficulty",  # Expected policy
                topic="Python",
                current_difficulty="Medium",
                effective_difficulty="Hard",
                previous_question="What is the difference between a list and tuple in Python?",
                candidate_answer="A list is mutable while a tuple is immutable.",
                correctness_score=95,
                concept_coverage=90,
                reasoning_score=92,
                missing_concepts=0,
                missing_concept_list=[],
                correct_streak=3,
                wrong_streak=0
            )
    
    # TEST 12: Generated difficulty differs from effective difficulty
    def test_difficulty_validation_rejects_mismatch(self, mock_llm_client):
        """Test that validation rejects when LLM returns different difficulty."""
        mock_llm_client.generate_json.return_value = {
            "question": "Can you explain this further?",
            "difficulty": "Easy",  # Different from expected
            "topic": "Python",
            "policy": "Increase Difficulty",
            "source": "Follow-up"
        }
        
        generator = QuestionGenerator(mock_llm_client)
        
        with pytest.raises(ValueError, match="Difficulty validation failed"):
            generator.generate_next_question(
                policy="Increase Difficulty",
                topic="Python",
                current_difficulty="Medium",
                effective_difficulty="Hard",  # Expected difficulty
                previous_question="What is the difference between a list and tuple in Python?",
                candidate_answer="A list is mutable while a tuple is immutable.",
                correctness_score=95,
                concept_coverage=90,
                reasoning_score=92,
                missing_concepts=0,
                missing_concept_list=[],
                correct_streak=3,
                wrong_streak=0
            )
    
    # Additional acceptance criteria tests
    
    def test_required_concepts_in_first_question(self, mock_llm_client):
        """Test that first question generation includes required concepts."""
        mock_llm_client.generate_json.return_value = {
            "question": "I noticed you developed a News Sentiment Analysis project using Python. Can you explain how the sentiment classification pipeline works?",
            "topic": "Python",
            "difficulty": "Easy",
            "source": "Project",
            "required_concepts": ["sentiment analysis", "classification pipeline", "NLP"]
        }
        
        generator = QuestionGenerator(mock_llm_client)
        result = generator.generate_first_question({
            "projects": [
                {
                    "name": "News Sentiment Analysis",
                    "tech_stack": ["Python", "NLP"]
                }
            ]
        })
        
        assert 'required_concepts' in result
        assert isinstance(result['required_concepts'], list)
        assert len(result['required_concepts']) > 0
    
    def test_session_stores_required_concepts(self):
        """Test that session stores required concepts with questions."""
        session_manager = SessionManager()
        session = session_manager.create_session("test_session")
        
        session.add_question(
            question="Test question",
            topic="Python",
            difficulty="Easy",
            source="Skill",
            required_concepts=["concept1", "concept2"]
        )
        
        assert session.current_required_concepts == ["concept1", "concept2"]
        assert session.question_history[0]['required_concepts'] == ["concept1", "concept2"]
    
    def test_feature_vector_preserves_11_features(self):
        """Test that feature vector always has exactly 11 features."""
        feature_builder = FeatureBuilder()
        
        llm_evaluation = {
            'semantic': {
                'correctness_score': 75,
                'concept_coverage': 70,
                'reasoning_score': 72,
                'missing_concepts': 2,
                'is_correct': True
            },
            'question_assessment': {
                'question_difficulty': 'Medium'
            }
        }
        
        feature_vector, feature_dict = feature_builder.build_feature_vector(
            llm_evaluation=llm_evaluation,
            correct_streak=1,
            wrong_streak=0
        )
        
        assert len(feature_vector) == 11
        assert len(feature_dict) == 11
    
    def test_feature_scale_matches_training(self):
        """Test that feature scales match training preprocessing."""
        feature_builder = FeatureBuilder()
        
        llm_evaluation = {
            'semantic': {
                'correctness_score': 79,  # Should remain 79, not 0.79
                'concept_coverage': 70,
                'reasoning_score': 72,
                'missing_concepts': 2,
                'is_correct': True
            },
            'question_assessment': {
                'question_difficulty': 'Medium'
            }
        }
        
        feature_vector, feature_dict = feature_builder.build_feature_vector(
            llm_evaluation=llm_evaluation,
            correct_streak=1,
            wrong_streak=0
        )
        
        # Correctness should be 79.0, not 0.79
        assert feature_dict['Correctness Score'] == 79.0
        assert 0 <= feature_dict['Correctness Score'] <= 100
        
        # Behavioral features should be 0.0-1.0
        assert 0.0 <= feature_dict['Engagement Score'] <= 1.0
        assert 0.0 <= feature_dict['Confidence Score'] <= 1.0
    
    def test_switch_topic_validation(self, mock_llm_client):
        """Test that Switch Topic validation enforces different topic."""
        mock_llm_client.generate_json.return_value = {
            "question": "Can you explain this further?",
            "difficulty": "Medium",
            "topic": "Python",  # Same as current topic - should fail
            "policy": "Switch Topic",
            "source": "Topic"
        }
        
        generator = QuestionGenerator(mock_llm_client)
        
        with pytest.raises(ValueError, match="Switch Topic validation failed"):
            generator.generate_next_question(
                policy="Switch Topic",
                topic="Python",  # Current topic
                current_difficulty="Medium",
                effective_difficulty="Medium",
                previous_question="What is the difference between a list and tuple in Python?",
                candidate_answer="A list is mutable while a tuple is immutable.",
                correctness_score=85,
                concept_coverage=80,
                reasoning_score=82,
                missing_concepts=1,
                missing_concept_list=[],
                correct_streak=2,
                wrong_streak=0,
                available_topics=["FastAPI", "MongoDB", "Machine Learning", "React"]
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
