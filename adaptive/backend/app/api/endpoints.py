import shutil
import logging
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
from app.services.parsing_service import ParsingService
from app.services.transcription_service import TranscriptionService
from app.services.tts_service import TTSService
from app.llm import InterviewEngine, FirstQuestionRequest, AnswerEvaluationRequest
from app.llm.llm_factory import create_llm_client
from app.interview.adaptive_orchestrator import AdaptiveInterviewOrchestrator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()
parsing_service = ParsingService()
transcription_service = TranscriptionService()
tts_service = TTSService()
interview_engine = InterviewEngine()

# Initialize adaptive orchestrator with LLM factory
try:
    import time
    import os
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("[ARCH] System Architecture Configuration")
    logger.info("=" * 70)
    logger.info(f"[ARCH] LLM_PROVIDER={os.getenv('LLM_PROVIDER', 'local_qwen')}")
    
    init_start = time.perf_counter()
    
    llm_client = create_llm_client()
    logger.info(f"[ARCH] LLM_CLIENT={type(llm_client).__name__}")
    logger.info(f"[ARCH] MODEL={os.getenv('LOCAL_QWEN_MODEL', 'Qwen/Qwen1.5-4B-Chat')}")
    logger.info(f"[ARCH] DEVICE={os.getenv('LOCAL_QWEN_DEVICE', 'auto')}")
    logger.info(f"[ARCH] QUANTIZATION={os.getenv('LOCAL_QWEN_QUANTIZATION', 'int4')}")
    logger.info(f"[ARCH] TABPFN_ENABLED=True")
    logger.info("=" * 70)
    logger.info("")
    
    # Force eager loading of LocalLLM model to avoid loading on first request
    if hasattr(llm_client, '_load_model'):
        try:
            logger.info("[PERF][STARTUP] Eagerly loading LocalLLM model...")
            llm_load_start = time.perf_counter()
            llm_client._load_model()
            llm_load_time = time.perf_counter() - llm_load_start
            logger.info(f"[PERF][STARTUP] LocalLLM model load: {llm_load_time*1000:.0f} ms")
            logger.info("[ARCH] LocalLLM model loaded and cached")
        except Exception as e:
            logger.warning(f"[API] Eager loading failed (will load on first request): {str(e)}")
    
    adaptive_orchestrator = AdaptiveInterviewOrchestrator(llm_client=llm_client)
    
    init_time = time.perf_counter() - init_start
    logger.info(f"[PERF][STARTUP] Total orchestrator initialization: {init_time*1000:.0f} ms")
    logger.info("[API] Adaptive orchestrator ready")
except Exception as e:
    logger.warning(f"[API] Failed to initialize adaptive orchestrator: {str(e)}")
    logger.warning(f"[API] Adaptive interview will not be available")
    adaptive_orchestrator = None

_LAST_PARSED_DATA = None


# Pydantic models for adaptive endpoints
class AdaptiveStartRequest(BaseModel):
    session_id: str
    resume_data: Dict[str, Any]
    voice: Optional[str] = "af_bella"


class AdaptiveAnswerRequest(BaseModel):
    session_id: str
    transcript: str
    audio_duration: Optional[float] = None
    voice: Optional[str] = "af_bella"

@router.post("/upload-resume")
async def upload_resume(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.pdf', '.docx', '.doc', '.txt')):
        raise HTTPException(status_code=400, detail="Only PDF, DOCX, and TXT files are supported.")

    uploads_dir = Path(__file__).parent.parent / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    
    saved_path = uploads_dir / file.filename
    with open(saved_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        global _LAST_PARSED_DATA
        result = parsing_service.parse_resume(saved_path, file.filename)
        _LAST_PARSED_DATA = result
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Parsing error: {str(e)}")

@router.get("/skills")
async def get_skills():
    if not _LAST_PARSED_DATA:
        raise HTTPException(status_code=404, detail="No resume uploaded yet.")
    return {
        "technical_skills": _LAST_PARSED_DATA.get("technical_skills", {})
    }

@router.get("/resume-summary")
async def get_resume_summary():
    if not _LAST_PARSED_DATA:
        raise HTTPException(status_code=404, detail="No resume uploaded yet.")
    return {
        "projects": _LAST_PARSED_DATA.get("projects", []),
        "technical_skills": _LAST_PARSED_DATA.get("technical_skills", {}),
        "area_of_interest": _LAST_PARSED_DATA.get("area_of_interest", []),
        "certifications": _LAST_PARSED_DATA.get("certifications", [])
    }

@router.get("/health")
async def health_check():
    return {"status": "ok", "service": "Resume Intelligence Engine", "version": "2.0.0"}

@router.post("/interview/start")
async def start_interview(request: FirstQuestionRequest):
    """
    Generate the first interview question based on resume data.
    """
    import time
    try:
        start_time = time.time()
        
        # Generate first question
        response = interview_engine.generate_first_question(
            resume_data=request.resume_data
        )
        
        total_time = time.time() - start_time
        logger.info(f"[API] /interview/start total time: {total_time:.2f}s")
        
        return {
            "question": response.question,
            "topic": response.topic,
            "difficulty": response.difficulty,
            "source": response.source
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start interview: {str(e)}")

@router.post("/interview/evaluate")
async def evaluate_answer(request: AnswerEvaluationRequest):
    """
    Evaluate a candidate's answer to an interview question.
    """
    try:
        # Evaluate the answer
        response = interview_engine.evaluate_answer(
            question=request.question,
            topic=request.topic,
            difficulty=request.difficulty,
            answer=request.answer
        )
        
        return {
            "correctness_score": response.evaluation.correctness_score,
            "concept_coverage": response.evaluation.concept_coverage,
            "reasoning_score": response.evaluation.reasoning_score,
            "missing_concepts": response.evaluation.missing_concepts,
            "is_correct": response.evaluation.is_correct,
            "difficulty": response.evaluation.difficulty,
            "feedback": response.evaluation.feedback
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to evaluate answer: {str(e)}")

@router.post("/interview/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Transcribe audio file to text using Whisper.
    """
    if not file.filename.lower().endswith(('.wav', '.mp3', '.m4a', '.ogg', '.webm')):
        raise HTTPException(status_code=400, detail="Only audio files are supported (wav, mp3, m4a, ogg, webm).")
    
    try:
        # Read audio file bytes
        audio_bytes = await file.read()
        
        logger.info(f"[STT][DEBUG] uploaded_audio_size: {len(audio_bytes)} bytes")
        logger.info(f"[STT][DEBUG] uploaded_mime_type: {file.content_type}")
        logger.info(f"[STT][DEBUG] file_extension: {file.filename.split('.')[-1] if '.' in file.filename else 'unknown'}")
        
        # Get file extension
        file_extension = file.filename.split('.')[-1] if '.' in file.filename else 'wav'
        
        # Transcribe audio
        transcript = transcription_service.transcribe_audio_bytes(
            audio_bytes=audio_bytes,
            file_extension=file_extension,
            language="en"
        )
        
        logger.info(f"[STT][DEBUG] whisper_raw_text: '{transcript}'")
        logger.info(f"[STT][DEBUG] final_transcript_length: {len(transcript)}")
        
        return {"transcript": transcript}
    except Exception as e:
        logger.error(f"[STT] Transcription failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

@router.post("/interview/tts")
async def text_to_speech(text: str, voice: str = "af_bella", speed: float = 1.0):
    """
    Convert text to speech using Kokoro TTS.
    
    Args:
        text: The text to synthesize.
        voice: Voice to use (default: 'af_bella').
        speed: Speech speed (0.5 to 2.0, default: 1.0).
    
    Returns:
        Audio file as streaming response.
    """
    import time
    try:
        start_time = time.time()
        
        # Synthesize speech
        audio_bytes = tts_service.synthesize(
            text=text,
            voice=voice,
            speed=speed
        )
        
        total_time = time.time() - start_time
        logger.info(f"[API] /interview/tts total time: {total_time:.2f}s")
        
        # Return as streaming response
        return StreamingResponse(
            iter([audio_bytes]),
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=speech.wav"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS failed: {str(e)}")


# Adaptive Interview Endpoints

@router.post("/interview/adaptive/start")
async def start_adaptive_interview(request: AdaptiveStartRequest):
    """
    Start a new adaptive interview session.
    
    Args:
        request: AdaptiveStartRequest with session_id and resume_data
    
    Returns:
        First question with TTS audio and session context
    """
    if not adaptive_orchestrator:
        raise HTTPException(status_code=503, detail="Adaptive orchestrator not available")
    
    try:
        result = adaptive_orchestrator.start_interview(
            session_id=request.session_id,
            resume_data=request.resume_data,
            voice=request.voice
        )
        
        # Return TTS audio as streaming response
        return StreamingResponse(
            iter([result['tts_audio']]),
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=question.wav",
                "X-Question-Number": str(result['question_number']),
                "X-Question": result['question'],
                "X-Topic": result['topic'],
                "X-Difficulty": result['difficulty'],
                "X-Source": result['source'],
                "X-Session-Id": result['session_id']
            }
        )
    except Exception as e:
        logger.error(f"[API] /interview/adaptive/start failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start adaptive interview: {str(e)}")


@router.post("/interview/adaptive/answer")
async def submit_adaptive_answer(request: AdaptiveAnswerRequest):
    """
    Submit a candidate's answer and get the next question.
    
    Args:
        request: AdaptiveAnswerRequest with session_id, transcript, and audio_duration
    
    Returns:
        Evaluation, policy, next question with TTS audio
    """
    print("\n" + "="*70)
    print("[DEBUG] submit_adaptive_answer CALLED")
    print(f"[DEBUG] Session ID: {request.session_id}")
    print(f"[DEBUG] Transcript: {request.transcript[:50] if request.transcript else 'None'}...")
    print("="*70 + "\n")
    
    # Validate transcript
    if not request.transcript or request.transcript.strip() == "":
        logger.warning(f"[API] Empty transcript received for session {request.session_id}")
        raise HTTPException(status_code=400, detail="Transcript cannot be empty")
    
    import time
    print(f"\n{'='*70}")
    print(f"[ANSWER REQUEST] Session: {request.session_id}")
    print(f"{'='*70}\n")
    logger.info(f"[PERF][ANSWER] request received for session {request.session_id}")
    answer_start = time.perf_counter()
    
    if not adaptive_orchestrator:
        raise HTTPException(status_code=503, detail="Adaptive orchestrator not available")
    
    try:
        result = adaptive_orchestrator.process_answer(
            session_id=request.session_id,
            transcript=request.transcript,
            audio_duration=request.audio_duration,
            voice=request.voice
        )
        
        answer_time = time.perf_counter() - answer_start
        logger.info(f"[PERF][ANSWER] backend processing completed: {answer_time:.3f}s")
        
        # If interview ended, return summary without audio
        if result.get('interview_ended'):
            return {
                'session_id': result['session_id'],
                'question_number': result['question_number'],
                'evaluation': result['evaluation'],
                'policy': result['policy'],
                'feature_vector': result['feature_vector'],
                'interview_ended': True,
                'session_summary': result['session_summary']
            }
        
        # Return TTS audio as streaming response
        return StreamingResponse(
            iter([result['tts_audio']]),
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=question.wav",
                "X-Question-Number": str(result['question_number']),
                "X-Question": result['next_question'],
                "X-Topic": result['next_topic'],
                "X-Difficulty": result['next_difficulty'],
                "X-Source": result['next_source'],
                "X-Session-Id": result['session_id'],
                "X-Policy": result['policy'],
                "X-Interview-Ended": "false"
            }
        )
    except Exception as e:
        logger.error(f"[API] /interview/adaptive/answer failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process answer: {str(e)}")


@router.post("/interview/adaptive/audio")
async def transcribe_adaptive_audio(session_id: str, file: UploadFile = File(...)):
    """
    Transcribe audio file to text using Whisper for adaptive interview.
    
    Args:
        session_id: The session identifier
        file: Audio file to transcribe
    
    Returns:
        Transcribed text
    """
    import time
    logger.info(f"[PERF][STT] request received for session {session_id}")
    stt_start = time.perf_counter()
    
    if not adaptive_orchestrator:
        raise HTTPException(status_code=503, detail="Adaptive orchestrator not available")
    
    if not file.filename.lower().endswith(('.wav', '.mp3', '.m4a', '.ogg', '.webm')):
        raise HTTPException(status_code=400, detail="Only audio files are supported (wav, mp3, m4a, ogg, webm).")
    
    try:
        # Read audio file bytes
        audio_bytes = await file.read()
        logger.info(f"[PERF][STT] audio bytes received: {len(audio_bytes)} bytes")
        
        # Get file extension
        file_extension = file.filename.split('.')[-1] if '.' in file.filename else 'wav'
        
        # Transcribe audio using orchestrator
        transcript = adaptive_orchestrator.transcribe_audio(
            audio_bytes=audio_bytes,
            file_extension=file_extension
        )
        
        stt_time = time.perf_counter() - stt_start
        logger.info(f"[PERF][STT] Whisper completed: {stt_time:.3f}s")
        logger.info(f"[PERF][STT] transcript length: {len(transcript)}")
        logger.info(f"[PERF][STT] response sent")
        
        return {
            "session_id": session_id,
            "transcript": transcript
        }
    except Exception as e:
        logger.error(f"[API] /interview/adaptive/audio failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


@router.get("/interview/adaptive/state")
async def get_adaptive_state(session_id: str):
    """
    Get the current state of an adaptive interview session.
    
    Args:
        session_id: The session identifier
    
    Returns:
        Current session state
    """
    if not adaptive_orchestrator:
        raise HTTPException(status_code=503, detail="Adaptive orchestrator not available")
    
    try:
        state = adaptive_orchestrator.get_session_state(session_id)
        return state
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"[API] /interview/adaptive/state failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get session state: {str(e)}")


@router.post("/interview/adaptive/end")
async def end_adaptive_interview(session_id: str):
    """
    End an adaptive interview session and return summary.
    
    Args:
        session_id: The session identifier
    
    Returns:
        Session summary
    """
    if not adaptive_orchestrator:
        raise HTTPException(status_code=503, detail="Adaptive orchestrator not available")
    
    try:
        summary = adaptive_orchestrator.end_session(session_id)
        return summary
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"[API] /interview/adaptive/end failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to end session: {str(e)}")
