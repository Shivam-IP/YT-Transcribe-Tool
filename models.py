from pydantic import BaseModel, HttpUrl
from typing import List, Optional
from enum import Enum


class WhisperModel(str, Enum):
    tiny = "tiny"
    base = "base"
    small = "small"
    medium = "medium"
    large = "large"


class CaptionRequest(BaseModel):
    url: str
    timestamps: bool = False
    whisper_model: WhisperModel = WhisperModel.base
    languages: List[str] = ["en", "en-US", "en-GB"]


class Segment(BaseModel):
    start: float
    end: Optional[float] = None
    duration: Optional[float] = None
    text: str


class CaptionResponse(BaseModel):
    source: str           # "youtube_captions" | "whisper_transcription"
    language: str
    text: str
    segments: List[Segment] = []
    video_id: str
    elapsed_seconds: Optional[float] = None   # total time taken


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None


class JobStatus(BaseModel):
    job_id: str
    status: str           # "pending" | "processing" | "done" | "failed"
    result: Optional[CaptionResponse] = None
    error: Optional[str] = None
    started_at: Optional[float] = None        # unix timestamp when job started
    elapsed_seconds: Optional[float] = None   # updated every poll