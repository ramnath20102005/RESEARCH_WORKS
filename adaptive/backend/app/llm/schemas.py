"""
Pydantic schemas for the Semantic Interview Engine.
Defines the data structures for interview requests and responses.
"""

from pydantic import BaseModel, Field
from typing import Optional


class FirstQuestionRequest(BaseModel):
    """Request schema for generating the first interview question."""
    
    resume_data: dict = Field(
        ...,
        description="Parsed resume JSON containing skills, projects, education, etc."
    )


class FirstQuestionResponse(BaseModel):
    """Response schema for the first interview question."""
    
    question: str = Field(..., description="The generated interview question")
    topic: str = Field(..., description="The main topic/skill being tested")
    difficulty: str = Field(default="Easy", description="Difficulty level of the question")
    source: str = Field(..., description="Source of the question: Project | Internship | Skill")


class AnswerEvaluationRequest(BaseModel):
    """Request schema for evaluating a candidate's answer."""
    
    question: str = Field(..., description="The interview question asked")
    topic: str = Field(..., description="The topic of the question")
    difficulty: str = Field(..., description="Difficulty level of the question")
    answer: str = Field(..., description="The candidate's transcribed answer")


class SemanticEvaluation(BaseModel):
    """Semantic evaluation scores for a candidate's answer."""
    
    correctness_score: int = Field(
        ...,
        ge=0,
        le=100,
        description="How correct the answer is (0-100)"
    )
    concept_coverage: int = Field(
        ...,
        ge=0,
        le=100,
        description="How well they covered relevant concepts (0-100)"
    )
    reasoning_score: int = Field(
        ...,
        ge=0,
        le=100,
        description="Quality of their reasoning and explanation (0-100)"
    )
    missing_concepts: int = Field(
        ...,
        ge=0,
        description="Number of important concepts they missed"
    )
    
    is_correct: bool = Field(..., description="Whether the answer is fundamentally correct")
    difficulty: str = Field(..., description="The difficulty level of the question")
    feedback: str = Field(..., description="Brief, constructive feedback")


class AnswerEvaluationResponse(BaseModel):
    """Response schema for answer evaluation."""
    
    evaluation: SemanticEvaluation = Field(..., description="The semantic evaluation results")
