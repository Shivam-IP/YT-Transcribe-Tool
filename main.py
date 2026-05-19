import uuid
from contextlib import asynccontextmanager
from typing import Dict

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from models import CaptionRequest, CaptionResponse, JobStatus, Segment
from services.youtube import extract_video_id, get_youtube_captions
from services.whisper import transcribe_from_url


# ── In-memory job store (swap with Redis for production) ─────────────────────
jobs: Dict[str, JobStatus] = {}


# ── App setup ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("YT Caption API started")
    yield
    print("YT Caption API shutting down")


app = FastAPI(
    title="YouTube Caption API",
    description="Extracts captions from YouTube — via transcript API or Whisper fallback",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Core caption logic ────────────────────────────────────────────────────────
def run_caption_pipeline(job_id: str, request: CaptionRequest):
    """Runs in background. Updates job store when done."""
    try:
        jobs[job_id].status = "processing"

        video_id = extract_video_id(request.url)

        # Step 1: Try YouTube captions
        result = get_youtube_captions(video_id, languages=request.languages)

        # Step 2: Fallback to Whisper
        if not result:
            print(f"[{job_id}] No YT captions — falling back to Whisper ({request.whisper_model})")
            result = transcribe_from_url(request.url, model_size=request.whisper_model.value)

        if not result:
            raise RuntimeError("Both caption methods failed")

        jobs[job_id].status = "done"
        jobs[job_id].result = CaptionResponse(
            source=result["source"],
            language=result["language"],
            text=result["text"],
            video_id=video_id,
            segments=[Segment(**s) for s in result["segments"]] if request.timestamps else [],
        )

    except Exception as e:
        jobs[job_id].status = "failed"
        jobs[job_id].error = str(e)
        print(f"[{job_id}] Failed: {e}")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return FileResponse("static/index.html")


@app.post("/captions", response_model=JobStatus, status_code=202)
def submit_caption_job(request: CaptionRequest, background_tasks: BackgroundTasks):
    """
    Submit a caption extraction job.
    Returns a job_id — poll /status/{job_id} for results.
    
    Why async/background?
    Whisper transcription can take 30s–3min depending on video length.
    We don't want the HTTP request to hang that long.
    """
    job_id = str(uuid.uuid4())
    jobs[job_id] = JobStatus(job_id=job_id, status="pending")
    background_tasks.add_task(run_caption_pipeline, job_id, request)
    return jobs[job_id]


@app.get("/status/{job_id}", response_model=JobStatus)
def get_job_status(job_id: str):
    """Poll this endpoint to check job progress and get results."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]


@app.post("/captions/sync", response_model=CaptionResponse)
def get_captions_sync(request: CaptionRequest):
    """
    Synchronous version — waits for result before responding.
    Fine for YouTube captions (fast). May timeout for long Whisper jobs.
    Use /captions (async) for unknown videos.
    """
    video_id = extract_video_id(request.url)

    result = get_youtube_captions(video_id, languages=request.languages)

    if not result:
        result = transcribe_from_url(request.url, model_size=request.whisper_model.value)

    if not result:
        raise HTTPException(status_code=422, detail="Could not extract captions via any method")

    return CaptionResponse(
        source=result["source"],
        language=result["language"],
        text=result["text"],
        video_id=video_id,
        segments=[Segment(**s) for s in result["segments"]] if request.timestamps else [],
    )


@app.get("/jobs", response_model=Dict[str, JobStatus])
def list_jobs():
    """List all jobs (dev convenience — remove in prod)."""
    return jobs


@app.delete("/jobs/{job_id}")
def delete_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    del jobs[job_id]
    return {"deleted": job_id}
