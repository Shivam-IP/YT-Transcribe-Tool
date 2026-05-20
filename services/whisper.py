import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional


def download_audio(video_url: str, output_dir: str) -> Optional[str]:
    """
    Download audio using yt-dlp CLI via subprocess.
    get-pot plugin works automatically in background.
    No cookies or manual token needed if plugin is installed.
    """
    output_template = os.path.join(output_dir, "audio.%(ext)s")
    cookies_path = os.path.expanduser("~/cookies.txt")

    command = [
        "yt-dlp",
        "-f", "bestaudio",
        "--extract-audio",
        "--audio-format", "mp3",
        "--audio-quality", "192K",
        "--ffmpeg-location", "/usr/bin",
        "-o", output_template,
    ]

    # Optional: use cookies if available (extra reliability)
    if os.path.exists(cookies_path):
        print(f"[yt-dlp] cookies found, adding for extra reliability")
        command += ["--cookies", cookies_path]

    command.append(video_url)

    print(f"[yt-dlp] running: {' '.join(command)}")

    result = subprocess.run(
        command,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"[yt-dlp] stderr: {result.stderr}")
        raise RuntimeError(
            f"yt-dlp failed (exit {result.returncode}).\n"
            f"Hint: {result.stderr.strip().splitlines()[-1] if result.stderr.strip() else 'unknown error'}\n\n"
            "Fix: pip install yt-dlp-get-pot bgutil-ytdlp-pot-provider"
        )

    # Find downloaded file
    for f in Path(output_dir).glob("audio.*"):
        print(f"[yt-dlp] downloaded: {f}")
        return str(f)

    return None


def transcribe_audio(audio_path: str, model_size: str = "base") -> Optional[dict]:
    """
    Transcribe audio using OpenAI Whisper (local, GPU if available).

    Model sizes:
      tiny   → fastest, rough accuracy
      base   → good default            ← default
      small  → better accuracy
      medium → near-human
      large  → best, slow on CPU
    """
    try:
        import whisper
    except ImportError:
        raise RuntimeError("Run: pip install openai-whisper")

    print(f"[whisper] loading model: {model_size}")
    model = whisper.load_model(model_size)

    print(f"[whisper] transcribing: {audio_path}")
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
    """Full pipeline: yt-dlp download → Whisper transcribe. Temp dir auto-cleans."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        audio_path = download_audio(video_url, tmp_dir)
        if not audio_path:
            raise RuntimeError("Audio download failed — check URL or yt-dlp setup")
        return transcribe_audio(audio_path, model_size=model_size)