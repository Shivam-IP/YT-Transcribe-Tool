import re
from typing import Optional


def extract_video_id(url: str) -> str:
    """Extract YouTube video ID from various URL formats."""
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
        r"(?:youtu\.be\/)([0-9A-Za-z_-]{11})",
        r"(?:embed\/)([0-9A-Za-z_-]{11})",
        r"(?:shorts\/)([0-9A-Za-z_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError(f"Could not extract video ID from: {url}")


def get_youtube_captions(video_id: str, languages: list = None) -> Optional[dict]:
    """
    Try to fetch captions directly from YouTube — no download needed.
    Priority: manual captions → auto-generated → any available language.
    Returns None if no captions are available.
    """
    try:
        from youtube_transcript_api import (
            YouTubeTranscriptApi,
            NoTranscriptFound,
            TranscriptsDisabled,
        )
    except ImportError:
        raise RuntimeError("Run: pip install youtube-transcript-api")

    languages = languages or ["en", "en-US", "en-GB"]

    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        transcript = None

        # Priority 1: manual captions in preferred language
        try:
            transcript = transcript_list.find_manually_created_transcript(languages)
            source_type = "manual"
        except NoTranscriptFound:
            pass

        # Priority 2: auto-generated captions
        if not transcript:
            try:
                transcript = transcript_list.find_generated_transcript(languages)
                source_type = "auto-generated"
            except NoTranscriptFound:
                pass

        # Priority 3: any language available
        if not transcript:
            for t in transcript_list:
                transcript = t
                source_type = "other-language"
                break

        if not transcript:
            return None

        segments = transcript.fetch()
        full_text = " ".join(s["text"] for s in segments)

        return {
            "source": "youtube_captions",   
            "language": transcript.language_code,
            "text": full_text,
            "segments": [
                {
                    "start": round(s["start"], 2),
                    "duration": round(s.get("duration", 0.0), 2),
                    "end": round(s["start"] + s.get("duration", 0.0), 2),
                    "text": s["text"],
                }
                for s in segments
            ],
        }

    except TranscriptsDisabled:
        return None
    except Exception as e:
        # Log and return None so caller falls back to Whisper
        print(f"[youtube] captions unavailable: {e}")
        return None
