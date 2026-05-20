import re
from typing import Optional


def extract_video_id(url: str) -> str:
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
    Fast path — fetch captions directly from YouTube.
    No download, no ffmpeg, no Whisper — pure HTTP call.
    Supports both new (>=0.7.x) and old (<0.7.x) API versions.
    Returns None if unavailable → caller falls back to yt-dlp + Whisper.
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError:
        raise RuntimeError("Run: pip install youtube-transcript-api")

    languages = languages or ["en", "en-US", "en-GB", "hi"]  # added Hindi too

    # ── New API (>= 0.7.x) ──────────────────────────────────────────
    try:
        api = YouTubeTranscriptApi()

        # Try preferred languages first
        try:
            fetched = api.fetch(video_id, languages=languages)
            segments = list(fetched)
            if segments:
                return _build_result(segments, languages[0], api_version="new")
        except Exception:
            pass

        # Fallback: fetch any available language
        try:
            fetched = api.fetch(video_id)
            segments = list(fetched)
            if segments:
                return _build_result(segments, "auto", api_version="new")
        except Exception:
            pass

    except Exception:
        pass

    # ── Old API (< 0.7.x) ───────────────────────────────────────────
    try:
        from youtube_transcript_api import NoTranscriptFound, TranscriptsDisabled

        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        transcript = None

        # Manual captions first
        try:
            transcript = transcript_list.find_manually_created_transcript(languages)
        except Exception:
            pass

        # Auto-generated captions
        if not transcript:
            try:
                transcript = transcript_list.find_generated_transcript(languages)
            except Exception:
                pass

        # Any available language
        if not transcript:
            for t in transcript_list:
                transcript = t
                break

        if transcript:
            segments = transcript.fetch()
            return _build_result(segments, transcript.language_code, api_version="old")

    except Exception as e:
        print(f"[youtube] old API failed: {e}")

    print(f"[youtube] no captions found for {video_id} — will fallback to Whisper")
    return None


def _build_result(segments, language: str, api_version: str) -> dict:
    """Build standardised result dict from segments (works for both API versions)."""

    def get_attr(seg, key, default=0.0):
        """Works for both object-style (new API) and dict-style (old API)."""
        if isinstance(seg, dict):
            return seg.get(key, default)
        return getattr(seg, key, default)

    result_segments = []
    for s in segments:
        start    = get_attr(s, "start")
        duration = get_attr(s, "duration")
        text     = get_attr(s, "text") or ""
        result_segments.append({
            "start":    round(start, 2),
            "duration": round(duration, 2),
            "end":      round(start + duration, 2),
            "text":     text,
        })

    full_text = " ".join(s["text"] for s in result_segments)

    return {
        "source":   "youtube_captions",
        "language": language,
        "text":     full_text,
        "segments": result_segments,
    }