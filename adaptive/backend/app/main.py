import sys
from pathlib import Path

# Automatically add app folder to Python module search path
app_dir = Path(__file__).parent
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.endpoints import router as api_router

app = FastAPI(
    title="Resume Intelligence & Skill Extraction API",
    description="Adaptive AI Interview System - Local Resume Parser & Skill Extractor (No LLM)",
    version="1.0.0"
)

# Add CORS middleware BEFORE including routers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

@app.get("/")
async def root():
    return {
        "service": "Resume Intelligence & Skill Extraction API",
        "status": "running",
        "docs": "http://127.0.0.1:8000/docs",
        "health": "http://127.0.0.1:8000/health"
    }

app.include_router(api_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
