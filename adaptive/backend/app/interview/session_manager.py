"""
Interview session manager for the Adaptive Interview System.

Maintains interview state including question history, context, streaks,
and provides context summary for LLM evaluation.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InterviewSession:
    """Represents a single interview session with a candidate."""
    
    def __init__(self, session_id: str, resume_data: Optional[Dict[str, Any]] = None):
        """
        Initialize an interview session.
        
        Args:
            session_id: Unique identifier for the session
            resume_data: Parsed resume data for the candidate
        """
        self.session_id = session_id
        self.resume_data = resume_data or {}
        self.created_at = datetime.now()
        
        # Interview state
        self.question_number = 0
        self.current_question: Optional[str] = None
        self.current_topic: Optional[str] = None
        self.current_difficulty: str = "Easy"  # Start with Easy
        
        # Streaks (calculated from is_correct history)
        self.correct_streak = 0
        self.wrong_streak = 0
        
        # History
        self.question_history: List[Dict[str, Any]] = []
        self.answer_history: List[Dict[str, Any]] = []
        self.evaluation_history: List[Dict[str, Any]] = []
        self.policy_history: List[Dict[str, Any]] = []
        
        # Session status
        self.is_active = True
        self.max_questions = 20  # Maximum questions per interview
        
        logger.info(f"[Session] Created session {session_id}")
    
    def add_question(
        self,
        question: str,
        topic: str,
        difficulty: str,
        source: str = "Skill"
    ):
        """
        Add a question to the session history.
        
        Args:
            question: The interview question
            topic: The question topic
            difficulty: The question difficulty (Easy/Medium/Hard)
            source: The source of the question (Project/Internship/Skill)
        """
        self.question_number += 1
        self.current_question = question
        self.current_topic = topic
        self.current_difficulty = difficulty
        
        self.question_history.append({
            'question_number': self.question_number,
            'question': question,
            'topic': topic,
            'difficulty': difficulty,
            'source': source,
            'timestamp': datetime.now().isoformat()
        })
        
        logger.info(f"[Session] Q{self.question_number}: {question[:50]}... (Difficulty: {difficulty})")
    
    def add_answer(
        self,
        transcript: str,
        audio_duration: Optional[float] = None
    ):
        """
        Add a candidate answer to the session history.
        
        Args:
            transcript: The transcribed answer text
            audio_duration: Duration of the audio recording in seconds
        """
        self.answer_history.append({
            'question_number': self.question_number,
            'transcript': transcript,
            'audio_duration': audio_duration,
            'timestamp': datetime.now().isoformat()
        })
        
        logger.info(f"[Session] A{self.question_number}: {transcript[:50]}...")
    
    def add_evaluation(self, evaluation: Dict[str, Any]):
        """
        Add an LLM evaluation to the session history.
        
        Args:
            evaluation: Dictionary containing semantic evaluation results
        """
        self.evaluation_history.append({
            'question_number': self.question_number,
            'evaluation': evaluation,
            'timestamp': datetime.now().isoformat()
        })
        
        # Update streaks based on is_correct
        is_correct = evaluation.get('semantic', {}).get('is_correct', False)
        
        if is_correct:
            self.correct_streak += 1
            self.wrong_streak = 0
        else:
            self.wrong_streak += 1
            self.correct_streak = 0
        
        # Update current difficulty from LLM assessment
        question_difficulty = evaluation.get('question_assessment', {}).get('question_difficulty')
        if question_difficulty:
            self.current_difficulty = question_difficulty
        
        logger.info(f"[Session] E{self.question_number}: is_correct={is_correct}, "
                    f"correct_streak={self.correct_streak}, wrong_streak={self.wrong_streak}")
    
    def add_policy(self, policy: str, reasoning: Optional[str] = None):
        """
        Add a TabPFN policy prediction to the session history.
        
        Args:
            policy: The predicted policy
            reasoning: Optional reasoning for the policy
        """
        self.policy_history.append({
            'question_number': self.question_number,
            'policy': policy,
            'reasoning': reasoning,
            'timestamp': datetime.now().isoformat()
        })
        
        logger.info(f"[Session] P{self.question_number}: Policy={policy}")
    
    def get_context_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current interview context for LLM evaluation.
        
        Returns:
            Dictionary containing current context state
        """
        return {
            'session_id': self.session_id,
            'question_number': self.question_number,
            'current_difficulty': self.current_difficulty,
            'current_topic': self.current_topic,
            'correct_streak': self.correct_streak,
            'wrong_streak': self.wrong_streak,
            'total_questions_asked': len(self.question_history),
            'is_active': self.is_active
        }
    
    def get_previous_evaluations(self, limit: int = 3) -> List[Dict[str, Any]]:
        """
        Get the most recent evaluations for context.
        
        Args:
            limit: Maximum number of previous evaluations to return
        
        Returns:
            List of recent evaluation dictionaries
        """
        return self.evaluation_history[-limit:] if self.evaluation_history else []
    
    def should_continue(self) -> bool:
        """
        Determine if the interview should continue.
        
        Returns:
            True if interview should continue, False otherwise
        """
        if not self.is_active:
            return False
        
        if self.question_number >= self.max_questions:
            logger.info(f"[Session] Max questions ({self.max_questions}) reached")
            return False
        
        return True
    
    def end_session(self):
        """Mark the session as ended."""
        self.is_active = False
        logger.info(f"[Session] Session {self.session_id} ended after {self.question_number} questions")
    
    def get_session_summary(self) -> Dict[str, Any]:
        """
        Get a complete summary of the interview session.
        
        Returns:
            Dictionary containing full session summary
        """
        # Calculate performance metrics
        total_evaluations = len(self.evaluation_history)
        if total_evaluations > 0:
            avg_correctness = sum(
                e['evaluation'].get('semantic', {}).get('correctness_score', 0)
                for e in self.evaluation_history
            ) / total_evaluations
            avg_coverage = sum(
                e['evaluation'].get('semantic', {}).get('concept_coverage', 0)
                for e in self.evaluation_history
            ) / total_evaluations
            avg_reasoning = sum(
                e['evaluation'].get('semantic', {}).get('reasoning_score', 0)
                for e in self.evaluation_history
            ) / total_evaluations
            correct_count = sum(
                1 for e in self.evaluation_history
                if e['evaluation'].get('semantic', {}).get('is_correct', False)
            )
        else:
            avg_correctness = 0
            avg_coverage = 0
            avg_reasoning = 0
            correct_count = 0
        
        return {
            'session_id': self.session_id,
            'created_at': self.created_at.isoformat(),
            'is_active': self.is_active,
            'total_questions': self.question_number,
            'max_correct_streak': max(
                (e['evaluation'].get('semantic', {}).get('is_correct', False)
                 for e in self.evaluation_history),
                default=0
            ),
            'avg_correctness': round(avg_correctness, 2),
            'avg_coverage': round(avg_coverage, 2),
            'avg_reasoning': round(avg_reasoning, 2),
            'correct_answers': correct_count,
            'accuracy': round(correct_count / total_evaluations * 100, 2) if total_evaluations > 0 else 0,
            'policy_distribution': self._get_policy_distribution()
        }
    
    def _get_policy_distribution(self) -> Dict[str, int]:
        """Get the distribution of policies used in the session."""
        distribution = {}
        for policy_record in self.policy_history:
            policy = policy_record['policy']
            distribution[policy] = distribution.get(policy, 0) + 1
        return distribution


class SessionManager:
    """Manages multiple interview sessions."""
    
    def __init__(self):
        """Initialize the session manager."""
        self.sessions: Dict[str, InterviewSession] = {}
        logger.info("[SessionManager] Initialized")
    
    def create_session(
        self,
        session_id: str,
        resume_data: Optional[Dict[str, Any]] = None
    ) -> InterviewSession:
        """
        Create a new interview session.
        
        Args:
            session_id: Unique identifier for the session
            resume_data: Parsed resume data for the candidate
        
        Returns:
            The created InterviewSession object
        """
        if session_id in self.sessions:
            logger.warning(f"[SessionManager] Session {session_id} already exists, returning existing")
            return self.sessions[session_id]
        
        session = InterviewSession(session_id, resume_data)
        self.sessions[session_id] = session
        return session
    
    def get_session(self, session_id: str) -> Optional[InterviewSession]:
        """
        Get an existing session by ID.
        
        Args:
            session_id: The session identifier
        
        Returns:
            The InterviewSession object if found, None otherwise
        """
        return self.sessions.get(session_id)
    
    def end_session(self, session_id: str) -> bool:
        """
        End an active session.
        
        Args:
            session_id: The session identifier
        
        Returns:
            True if session was ended, False if not found
        """
        session = self.get_session(session_id)
        if session:
            session.end_session()
            return True
        return False
    
    def get_all_sessions(self) -> List[InterviewSession]:
        """Get all active sessions."""
        return list(self.sessions.values())
    
    def cleanup_inactive_sessions(self):
        """Remove inactive sessions from memory."""
        inactive_ids = [
            session_id for session_id, session in self.sessions.items()
            if not session.is_active
        ]
        for session_id in inactive_ids:
            del self.sessions[session_id]
            logger.info(f"[SessionManager] Cleaned up inactive session {session_id}")
