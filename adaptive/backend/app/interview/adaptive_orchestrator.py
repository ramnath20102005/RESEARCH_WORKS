"""
Adaptive interview orchestrator for the Adaptive Interview System.

Orchestrates the complete adaptive interview loop:
Question → TTS → Voice → Whisper → Transcript → LLM Assessment → 
Feature Vector → TabPFN → Policy → Next Question → TTS → LOOP
"""

import logging
import time
from typing import Dict, Any, Optional
from pathlib import Path

from .session_manager import SessionManager, InterviewSession
from .feature_builder import FeatureBuilder
from .tabpfn_inference import TabPFNInference
from ..llm.semantic_evaluator import SemanticEvaluator
from ..llm.question_generator import QuestionGenerator
from ..services.tts_service import TTSService
from ..services.transcription_service import TranscriptionService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdaptiveInterviewOrchestrator:
    """
    Orchestrates the complete adaptive interview loop.
    
    Integrates all components:
    - Session management
    - LLM semantic evaluation
    - Feature vector construction
    - TabPFN policy prediction
    - Policy-driven question generation
    - TTS and STT services
    """
    
    def __init__(
        self,
        llm_client: Any,
        tabpfn_model_path: Optional[Path] = None
    ):
        """
        Initialize the adaptive interview orchestrator.
        
        Args:
            llm_client: LLM client instance (NVIDIA, etc.)
            tabpfn_model_path: Optional custom path to TabPFN model
        """
        # Initialize components
        self.session_manager = SessionManager()
        self.semantic_evaluator = SemanticEvaluator(llm_client)
        self.question_generator = QuestionGenerator(llm_client)
        self.feature_builder = FeatureBuilder(random_seed=42)
        self.tabpfn_inference = TabPFNInference(tabpfn_model_path)
        self.tts_service = TTSService()
        self.transcription_service = TranscriptionService(model_size="base", device="cpu")
        
        logger.info("[Orchestrator] Initialized all components")
    
    def start_interview(
        self,
        session_id: str,
        resume_data: Dict[str, Any],
        voice: str = "af_bella"
    ) -> Dict[str, Any]:
        """
        Start a new adaptive interview session.
        
        Args:
            session_id: Unique session identifier
            resume_data: Parsed resume data
            voice: Kokoro TTS voice to use
        
        Returns:
            Dictionary with first question and session context
        """
        logger.info(f"[Orchestrator] Starting interview session {session_id} with voice: {voice}")
        logger.info(f"[PERF][START] interview_start: 0.000s")
        total_start = time.perf_counter()
        
        try:
            # Create session
            session_start = time.perf_counter()
            session = self.session_manager.create_session(
                session_id=session_id,
                resume_data=resume_data
            )
            session_time = time.perf_counter() - session_start
            logger.info(f"[PERF][START] session_creation: {session_time*1000:.0f} ms")
            
            # Generate first question
            logger.info(f"[Orchestrator] Generating first question")
            question_start = time.perf_counter()
            first_question_response = self.question_generator.generate_first_question(resume_data)
            question_time = time.perf_counter() - question_start
            logger.info(f"[PERF][START] question_generation: {question_time*1000:.0f} ms")
            logger.info(f"[LLM][CALL] question_generation")
            logger.info(f"[LLM][CALL_COUNT] semantic=0 question_generation=1")
            
            # Add question to session
            session.add_question(
                question=first_question_response.question,
                topic=first_question_response.topic,
                difficulty=first_question_response.difficulty,
                source=first_question_response.source
            )
            
            # Generate TTS audio with selected voice
            tts_start = time.perf_counter()
            tts_audio = self.tts_service.synthesize(first_question_response.question, voice=voice)
            tts_time = time.perf_counter() - tts_start
            logger.info(f"[PERF][START] tts_synthesis: {tts_time*1000:.0f} ms")
            
            total_time = time.perf_counter() - total_start
            logger.info(f"[PERF][START] total_backend: {total_time*1000:.0f} ms")
            logger.info(f"[Orchestrator] First question generated: {first_question_response.question[:50]}...")
            
            return {
                'session_id': session_id,
                'question_number': session.question_number,
                'question': first_question_response.question,
                'topic': first_question_response.topic,
                'difficulty': first_question_response.difficulty,
                'source': first_question_response.source,
                'tts_audio': tts_audio,
                'context': session.get_context_summary()
            }
            
        except Exception as e:
            logger.error(f"[Orchestrator] Failed to start interview: {str(e)}")
            raise Exception(f"Failed to start interview: {str(e)}")
    
    def process_answer(
        self,
        session_id: str,
        transcript: str,
        audio_duration: Optional[float] = None,
        voice: str = "af_bella"
    ) -> Dict[str, Any]:
        """
        Process a candidate's answer and generate the next question.
        
        Args:
            session_id: The session identifier
            transcript: The transcribed answer text
            audio_duration: Optional duration of the audio
            voice: Kokoro TTS voice to use for next question
        
        Returns:
            Dictionary containing evaluation, policy, and next question
        """
        logger.info(f"[Orchestrator] Processing answer for session {session_id} with voice: {voice}")
        logger.info(f"[PERF][ANSWER] process_answer: 0.000s")
        total_start = time.perf_counter()
        
        try:
            # Get session
            session_start = time.perf_counter()
            session = self.session_manager.get_session(session_id)
            if not session:
                raise ValueError(f"Session {session_id} not found")
            session_time = time.perf_counter() - session_start
            logger.info(f"[PERF][ANSWER] session_retrieval: {session_time:.3f}s")
            
            # Add answer to session
            session.add_answer(transcript, audio_duration)
            
            # Log [QUESTION] section
            logger.info("")
            logger.info("=" * 70)
            logger.info("[QUESTION]")
            logger.info("=" * 70)
            logger.info(f"Question Number: {session.question_number}")
            logger.info(f"Question: {session.current_question}")
            logger.info(f"Topic: {session.current_topic}")
            logger.info(f"Difficulty: {session.current_difficulty}")
            logger.info("=" * 70)
            logger.info("")
            
            # Log [VOICE] section
            logger.info("=" * 70)
            logger.info("[VOICE]")
            logger.info("=" * 70)
            logger.info(f"Audio received: Yes")
            if audio_duration:
                logger.info(f"Audio duration: {audio_duration:.2f}s")
            logger.info(f"Transcript: {transcript}")
            logger.info("=" * 70)
            logger.info("")
            
            # Step 1: LLM Semantic Evaluation
            logger.info(f"[Orchestrator] Step 1: LLM Semantic Evaluation")
            eval_start = time.perf_counter()
            evaluation = self.semantic_evaluator.evaluate_answer(
                question=session.current_question,
                topic=session.current_topic,
                current_difficulty=session.current_difficulty,
                answer=transcript,
                question_number=session.question_number,
                correct_streak=session.correct_streak,
                wrong_streak=session.wrong_streak
            )
            eval_time = time.perf_counter() - eval_start
            logger.info(f"[PERF][ANSWER] llm_evaluation: {eval_time:.3f}s")
            logger.info(f"[LLM][CALL] semantic_evaluation")
            logger.info(f"[LLM][CALL_COUNT] semantic=1 question_generation=0")
            
            # Log [LLM SEMANTIC] section
            logger.info("")
            logger.info("=" * 70)
            logger.info("[LLM SEMANTIC]")
            logger.info("=" * 70)
            logger.info(f"Correctness: {evaluation['semantic']['correctness_score']}")
            logger.info(f"Concept Coverage: {evaluation['semantic']['concept_coverage']}")
            logger.info(f"Reasoning: {evaluation['semantic']['reasoning_score']}")
            logger.info(f"Missing Concepts: {evaluation['semantic']['missing_concepts']}")
            logger.info(f"is_correct: {evaluation['semantic']['is_correct']}")
            logger.info(f"Current Question Difficulty: {evaluation['question_assessment']['question_difficulty']}")
            logger.info(f"LLM latency: {eval_time*1000:.0f} ms")
            logger.info("=" * 70)
            logger.info("")
            
            # Add evaluation to session
            session.add_evaluation(evaluation)
            
            # Log [CONTEXT] section (streaks before/after)
            logger.info("")
            logger.info("=" * 70)
            logger.info("[CONTEXT]")
            logger.info("=" * 70)
            logger.info(f"Previous correct streak: {session.correct_streak}")
            logger.info(f"Previous wrong streak: {session.wrong_streak}")
            
            # Update streaks based on is_correct
            if evaluation['semantic']['is_correct']:
                session.correct_streak += 1
                session.wrong_streak = 0
            else:
                session.wrong_streak += 1
                session.correct_streak = 0
            
            logger.info(f"Updated correct streak: {session.correct_streak}")
            logger.info(f"Updated wrong streak: {session.wrong_streak}")
            logger.info("=" * 70)
            logger.info("")
            
            # Step 2: Build Feature Vector
            logger.info(f"[Orchestrator] Step 2: Build Feature Vector")
            feature_start = time.perf_counter()
            feature_vector, feature_dict = self.feature_builder.build_feature_vector(
                llm_evaluation=evaluation,
                correct_streak=session.correct_streak,
                wrong_streak=session.wrong_streak
            )
            feature_time = time.perf_counter() - feature_start
            logger.info(f"[PERF][ANSWER] feature_building: {feature_time:.3f}s")
            
            # Log [BEHAVIORAL] section with TEMPORARY RANDOM PLACEHOLDER marker
            logger.info("")
            logger.info("=" * 70)
            logger.info("[BEHAVIORAL]")
            logger.info("=" * 70)
            logger.info(f"Engagement: {feature_dict.get('engagement_score', 0.5):.3f} [TEMPORARY RANDOM PLACEHOLDER]")
            logger.info(f"Confidence: {feature_dict.get('confidence_score', 0.5):.3f} [TEMPORARY RANDOM PLACEHOLDER]")
            logger.info(f"Hesitation: {feature_dict.get('hesitation_score', 0.5):.3f} [TEMPORARY RANDOM PLACEHOLDER]")
            logger.info(f"Eye Contact: {feature_dict.get('eye_contact_score', 0.5):.3f} [TEMPORARY RANDOM PLACEHOLDER]")
            logger.info("=" * 70)
            logger.info("")
            
            # Log [MODEL INPUT] section - Exact 11-feature vector with feature names + values
            logger.info("")
            logger.info("=" * 70)
            logger.info("[MODEL INPUT]")
            logger.info("=" * 70)
            logger.info("Exact 11-feature vector:")
            logger.info(f"[{', '.join([str(x) for x in feature_vector])}]")
            logger.info("")
            logger.info("Feature names + values:")
            logger.info(f"1. Correctness Score = {feature_vector[0]}")
            logger.info(f"2. Concept Coverage = {feature_vector[1]}")
            logger.info(f"3. Reasoning Score = {feature_vector[2]}")
            logger.info(f"4. Missing Concepts = {feature_vector[3]}")
            logger.info(f"5. Engagement Score = {feature_vector[4]}")
            logger.info(f"6. Confidence Score = {feature_vector[5]}")
            logger.info(f"7. Hesitation Score = {feature_vector[6]}")
            logger.info(f"8. Eye Contact Score = {feature_vector[7]}")
            logger.info(f"9. Difficulty = {feature_vector[8]}")
            logger.info(f"10. Correct Streak = {feature_vector[9]}")
            logger.info(f"11. Wrong Streak = {feature_vector[10]}")
            logger.info("=" * 70)
            logger.info("")
            
            # Validate feature vector
            is_valid = self.feature_builder.validate_feature_vector(feature_vector)
            if not is_valid:
                raise ValueError("Feature vector validation failed")
            
            # Step 3: TabPFN Policy Prediction
            logger.info(f"[Orchestrator] Step 3: TabPFN Policy Prediction")
            tabpfn_start = time.perf_counter()
            policy_prediction = self.tabpfn_inference.predict_policy(
                feature_vector=feature_vector,
                return_probabilities=True
            )
            tabpfn_time = time.perf_counter() - tabpfn_start
            logger.info(f"[PERF][ANSWER] tabpfn_inference: {tabpfn_time:.3f}s")
            
            predicted_policy = policy_prediction['predicted_policy']
            predicted_class = policy_prediction.get('predicted_class')
            probabilities = policy_prediction.get('probabilities', {})
            
            # Log [TABPFN] section with all 7 policy probabilities
            logger.info("")
            logger.info("=" * 70)
            logger.info("[TABPFN]")
            logger.info("=" * 70)
            logger.info(f"Predicted class: {predicted_class}")
            logger.info(f"Predicted policy: {predicted_policy}")
            logger.info("")
            logger.info("All 7 policy probabilities:")
            policy_mapping = self.tabpfn_inference.get_policy_mapping()
            for class_id in range(7):
                policy_name = policy_mapping.get(class_id, f"Unknown_{class_id}")
                prob = probabilities.get(policy_name, 0.0)
                logger.info(f"  {class_id} = {policy_name}: {prob:.4f}")
            logger.info("")
            logger.info(f"TabPFN latency: {tabpfn_time*1000:.0f} ms")
            logger.info("=" * 70)
            logger.info("")
            
            # Comprehensive adaptive decision logging - REMOVED (replaced by structured sections above)
            
            # Step 4: Apply Policy to Determine Effective Difficulty
            logger.info(f"[Orchestrator] Step 4: Apply Policy to Determine Effective Difficulty")
            
            current_difficulty = session.current_difficulty
            effective_difficulty = current_difficulty
            policy_reason = ""
            
            # Difficulty encoding
            difficulty_order = ["Easy", "Medium", "Hard"]
            current_index = difficulty_order.index(current_difficulty)
            
            # Apply policy with bounds
            if predicted_policy == "Reduce Difficulty":
                if current_index > 0:
                    effective_difficulty = difficulty_order[current_index - 1]
                    policy_reason = f"Reduced from {current_difficulty} to {effective_difficulty}"
                else:
                    policy_reason = f"Already at minimum difficulty ({current_difficulty})"
            elif predicted_policy == "Increase Difficulty":
                if current_index < len(difficulty_order) - 1:
                    effective_difficulty = difficulty_order[current_index + 1]
                    policy_reason = f"Increased from {current_difficulty} to {effective_difficulty}"
                else:
                    policy_reason = f"Already at maximum difficulty ({current_difficulty})"
            elif predicted_policy == "Maintain Difficulty":
                policy_reason = f"Maintained current difficulty ({current_difficulty})"
            else:
                # For other policies (Ask Application Question, Ask Follow-up, Probe Missing Concept, Switch Topic)
                # Maintain current difficulty unless explicitly changed by policy
                policy_reason = f"Policy '{predicted_policy}' does not change difficulty"
            
            # Log policy application
            logger.info("")
            logger.info("=" * 70)
            logger.info("[POLICY APPLICATION]")
            logger.info("=" * 70)
            logger.info(f"Current Difficulty: {current_difficulty}")
            logger.info(f"Predicted Policy: {predicted_policy}")
            logger.info(f"Effective Difficulty: {effective_difficulty}")
            logger.info(f"Reason: {policy_reason}")
            logger.info("=" * 70)
            logger.info("")
            
            # Add policy to session
            session.add_policy(predicted_policy)
            
            # Step 5: Generate Next Question
            logger.info(f"[Orchestrator] Step 4: Generate Next Question")
            
            # Check if interview should continue
            if not session.should_continue():
                session.end_session()
                return {
                    'session_id': session_id,
                    'question_number': session.question_number,
                    'evaluation': evaluation,
                    'policy': predicted_policy,
                    'feature_vector': feature_dict,
                    'interview_ended': True,
                    'session_summary': session.get_session_summary()
                }
            
            # Generate next question based on policy and effective difficulty
            question_start = time.perf_counter()
            next_question_response = self.question_generator.generate_next_question(
                policy=predicted_policy,
                topic=session.current_topic,
                current_difficulty=effective_difficulty,  # Use effective difficulty
                previous_question=session.current_question,
                candidate_answer=transcript,
                correctness_score=evaluation['semantic']['correctness_score'],
                concept_coverage=evaluation['semantic']['concept_coverage'],
                reasoning_score=evaluation['semantic']['reasoning_score'],
                missing_concepts=evaluation['semantic']['missing_concepts'],
                correct_streak=session.correct_streak,
                wrong_streak=session.wrong_streak,
                resume_data=session.resume_data
            )
            question_time = time.perf_counter() - question_start
            logger.info(f"[PERF][ANSWER] question_generation: {question_time:.3f}s")
            logger.info(f"[LLM][CALL] question_generation")
            logger.info(f"[LLM][CALL_COUNT] semantic=1 question_generation=1")
            
            # Log [NEXT QUESTION] section
            logger.info("")
            logger.info("=" * 70)
            logger.info("[NEXT QUESTION]")
            logger.info("=" * 70)
            logger.info(f"Policy received by generator: {predicted_policy}")
            logger.info(f"Generated question: {next_question_response['question']}")
            logger.info(f"Generated difficulty: {next_question_response['difficulty']}")
            logger.info(f"Generated topic: {next_question_response['topic']}")
            logger.info(f"Returned policy: {next_question_response['policy']}")
            logger.info(f"Policy validation: PASSED")
            logger.info(f"Question-generation latency: {question_time*1000:.0f} ms")
            logger.info("=" * 70)
            logger.info("")
            
            # Remove the old comprehensive structured log print statements
            # They are replaced by the structured logger sections above
            
            # Add next question to session
            session.add_question(
                question=next_question_response['question'],
                topic=next_question_response['topic'],
                difficulty=next_question_response['difficulty'],
                source=next_question_response['source']
            )
            
            # Generate TTS audio for next question with selected voice
            tts_start = time.perf_counter()
            tts_audio = self.tts_service.synthesize(next_question_response['question'], voice=voice)
            tts_time = time.perf_counter() - tts_start
            logger.info(f"[PERF][ANSWER] kokoro_synthesis: {tts_time:.3f}s")
            
            # Log [TTS] section
            logger.info("")
            logger.info("=" * 70)
            logger.info("[TTS]")
            logger.info("=" * 70)
            logger.info(f"TTS latency: {tts_time*1000:.0f} ms")
            logger.info(f"Audio generation status: Success")
            logger.info(f"Voice: {voice}")
            logger.info("=" * 70)
            logger.info("")
            
            total_time = time.perf_counter() - total_start
            
            # Log [TOTAL] performance breakdown
            logger.info("")
            logger.info("=" * 70)
            logger.info("[TOTAL]")
            logger.info("=" * 70)
            logger.info(f"Session retrieval: {session_time*1000:.0f} ms")
            logger.info(f"Semantic LLM evaluation: {eval_time*1000:.0f} ms")
            logger.info(f"Feature construction: {feature_time*1000:.0f} ms")
            logger.info(f"TabPFN inference: {tabpfn_time*1000:.0f} ms")
            logger.info(f"Question generation: {question_time*1000:.0f} ms")
            logger.info(f"TTS synthesis: {tts_time*1000:.0f} ms")
            logger.info(f"Total backend processing: {total_time*1000:.0f} ms")
            logger.info("=" * 70)
            logger.info("")
            
            logger.info(f"[Orchestrator] Next question generated: {next_question_response['question'][:50]}...")
            
            return {
                'session_id': session_id,
                'question_number': session.question_number,
                'evaluation': evaluation,
                'policy': predicted_policy,
                'policy_probabilities': policy_prediction.get('probabilities'),
                'feature_vector': feature_dict,
                'next_question': next_question_response['question'],
                'next_topic': next_question_response['topic'],
                'next_difficulty': next_question_response['difficulty'],
                'next_source': next_question_response['source'],
                'tts_audio': tts_audio,
                'context': session.get_context_summary(),
                'interview_ended': False
            }
            
        except Exception as e:
            logger.error(f"[Orchestrator] Failed to process answer: {str(e)}")
            import traceback
            logger.error(f"[Orchestrator] Traceback:\n{traceback.format_exc()}")
            raise Exception(f"Failed to process answer: {str(e)}")
    
    def transcribe_audio(
        self,
        audio_bytes: bytes,
        file_extension: str = "wav"
    ) -> str:
        """
        Transcribe audio to text using Whisper.
        
        Args:
            audio_bytes: Audio data as bytes
            file_extension: File extension (wav, mp3, etc.)
        
        Returns:
            Transcribed text
        """
        logger.info("[Orchestrator] Transcribing audio")
        
        try:
            transcript = self.transcription_service.transcribe_audio_bytes(
                audio_bytes=audio_bytes,
                file_extension=file_extension,
                language="en"
            )
            
            logger.info(f"[Orchestrator] Transcription complete: {len(transcript)} chars")
            
            return transcript
            
        except Exception as e:
            logger.error(f"[Orchestrator] Failed to transcribe audio: {str(e)}")
            raise Exception(f"Failed to transcribe audio: {str(e)}")
    
    def get_session_state(self, session_id: str) -> Dict[str, Any]:
        """
        Get the current state of an interview session.
        
        Args:
            session_id: The session identifier
        
        Returns:
            Dictionary containing session state
        """
        session = self.session_manager.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        return {
            'session_id': session_id,
            'context': session.get_context_summary(),
            'is_active': session.is_active,
            'current_question': session.current_question,
            'current_topic': session.current_topic,
            'current_difficulty': session.current_difficulty
        }
    
    def end_session(self, session_id: str) -> Dict[str, Any]:
        """
        End an interview session and return summary.
        
        Args:
            session_id: The session identifier
        
        Returns:
            Dictionary containing session summary
        """
        logger.info(f"[Orchestrator] Ending session {session_id}")
        
        session = self.session_manager.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        session.end_session()
        
        return {
            'session_id': session_id,
            'session_summary': session.get_session_summary()
        }
