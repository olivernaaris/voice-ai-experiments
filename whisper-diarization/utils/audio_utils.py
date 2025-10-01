from __future__ import annotations

import base64
import subprocess
import os
import requests
import tempfile
import shutil
from typing import Tuple


def get_file(file_path=None, file_url=None, file_string=None) -> Tuple[str, str]:
    """
    Handles any input type and converts it to PCM 16kHz WAV.
    Returns the path to the converted file and temp directory.
    """
    if not any([file_path, file_url, file_string]):
        raise ValueError("One of file_path, file_url, or file_string must be provided.")

    temp_dir = tempfile.mkdtemp()
    raw_path = os.path.join(temp_dir, "input_raw")
    processed_path = os.path.join(temp_dir, "input_pcm16.wav")

    # Save raw input
    if file_path:
        shutil.copy(file_path, raw_path)
    elif file_url:
        r = requests.get(file_url, timeout=60)
        r.raise_for_status()
        with open(raw_path, "wb") as f:
            f.write(r.content)
    elif file_string:
        audio_bytes = base64.b64decode(
            file_string.split(",")[1] if "," in file_string else file_string
        )
        with open(raw_path, "wb") as f:
            f.write(audio_bytes)

    # Convert to PCM 16kHz
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            raw_path,
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            processed_path,
        ],
        check=True,
    )

    return processed_path, temp_dir


def get_audio_channels(file_path: str) -> int:
    """Identify the number of audio channels using ffprobe."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=channels",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            file_path,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
    return int(result.stdout.strip())


def split_stereo_channels(file_path: str, temp_dir: str) -> Tuple[str, str]:
    """Splits stereo audio into two mono files (left and right channel)."""
    ch1_path = os.path.join(temp_dir, "channel1.wav")
    ch2_path = os.path.join(temp_dir, "channel2.wav")

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            file_path,
            "-map_channel",
            "0.0.0",
            "-acodec",
            "pcm_s16le",
            "-ac",
            "1",
            "-ar",
            "16000",
            ch1_path,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            file_path,
            "-map_channel",
            "0.0.1",
            "-acodec",
            "pcm_s16le",
            "-ac",
            "1",
            "-ar",
            "16000",
            ch2_path,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    return ch1_path, ch2_path
