import os
import tempfile
from pathlib import Path
from typing import Optional


def download_audio(video_url: str, output_dir: str) -> Optional[str]:
    """
    Download ONLY the audio stream using yt-dlp (no video).
    Converts to mp3 via ffmpeg. Returns path to audio file.
    """
    try:
        import yt_dlp
    except ImportError:
        raise RuntimeError("Run: pip install yt-dlp")

    output_template = os.path.join(output_dir, "audio.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])

    # Find downloaded file
    for f in Path(output_dir).glob("audio.*"):
        return str(f)

    return None


def transcribe_audio(audio_path: str, model_size: str = "base") -> Optional[dict]:
    """
    Transcribe audio using OpenAI Whisper (runs locally, uses GPU if available).

    Model size guide:
      tiny   → ~1GB RAM, fastest, rough accuracy
      base   → ~1GB RAM, good for most cases       ← default
      small  → ~2GB RAM, better accuracy
      medium → ~5GB RAM, near-human accuracy
      large  → ~10GB RAM, best (slow on CPU)
    """
    try:
        import whisper
    except ImportError:
        raise RuntimeError("Run: pip install openai-whisper")

    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path, verbose=False)

    return {
        "source": "whisper_transcription",
        "language": result.get("language", "unknown"),
        "text": result["text"].strip(),
        "segments": [
            {
                "start": round(seg["start"], 2),
                "end": round(seg["end"], 2),
                "duration": round(seg["end"] - seg["start"], 2),
                "text": seg["text"].strip(),
            }
            for seg in result.get("segments", [])
        ],
    }


def transcribe_from_url(video_url: str, model_size: str = "base") -> Optional[dict]:
    """
    Full pipeline: download audio → transcribe.
    Uses a temp directory that auto-cleans up.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        audio_path = download_audio(video_url, tmp_dir)
        if not audio_path:
            raise RuntimeError("Audio download failed — check the URL or yt-dlp version")
        return transcribe_audio(audio_path, model_size=model_size)
