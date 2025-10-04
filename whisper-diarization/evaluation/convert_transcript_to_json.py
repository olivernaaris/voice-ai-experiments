#!/usr/bin/env uv run
"""Convert plain text transcript to JSON format for ASR evaluation.

This utility converts transcripts with speaker labels and timestamps
into the JSON format required by test_asr_evaluation.py.

Usage:
    ./convert_transcript_to_json.py --input transcript.txt --output reference.json
"""

import json
import re
from pathlib import Path

import typer

app = typer.Typer(help="Convert plain text transcripts to JSON format")


# Estonian filler words and patterns
FILLER_PATTERNS = [
    # Interjections and acknowledgments
    r"\bMhm\b",
    r"\bmhm\b",
    r"\bAa+\b",  # Aa, Aaa, etc.
    r"\baa+\b",  # aa, aaa, etc.
    # Tag questions and discourse markers
    r"\bvä\b",  # colloquial "isn't it?"
    r"\bonju\b",  # tag question "right?"
    # Discourse markers (when standalone or at boundaries)
    r"\bnoh\b",  # well
    r"\bNo\b",  # well/so (at start of sentence)
    # "nagu" as filler (like/you know)
    r"\bnagu\b",
    # "jah" when clearly filler (at end after comma or multiple times)
    r",\s*jah\b",
    r"\bjah,",
    r"\bjah\s+jah\b",
    # Repetitions (word repeated immediately)
    r"\b(\w+)\s+\1\b",  # catches "see see", "ta ta", "ja ja", etc.
]


def clean_fillers(text: str) -> str:
    """Remove Estonian filler words from text while preserving meaningful content.

    Args:
        text: Original text with potential fillers

    Returns:
        Cleaned text with fillers removed
    """
    cleaned = text

    # Remove filler patterns
    for pattern in FILLER_PATTERNS:
        # Special handling for repetition pattern
        if pattern == r"\b(\w+)\s+\1\b":
            # Replace repeated words with single instance
            cleaned = re.sub(pattern, r"\1", cleaned, flags=re.IGNORECASE)
        else:
            # Remove the filler word
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

    # Clean up extra spaces and punctuation
    cleaned = re.sub(r"\s+", " ", cleaned)  # Multiple spaces to single space
    cleaned = re.sub(r"\s+([.,!?])", r"\1", cleaned)  # Space before punctuation
    cleaned = re.sub(r"([.,!?])\s*\1+", r"\1", cleaned)  # Duplicate punctuation
    cleaned = re.sub(r"^[.,\s]+", "", cleaned)  # Leading punctuation/space
    cleaned = re.sub(r"[.,\s]+$", "", cleaned)  # Trailing punctuation/space

    # Ensure sentence ends with period if it had content
    if cleaned and cleaned[-1] not in ".!?":
        cleaned += "."

    return cleaned.strip()


def parse_transcript(content: str, remove_fillers: bool = False) -> list[dict]:
    """Parse transcript with speaker labels and timestamps into segments.

    Args:
        content: Raw transcript text with format like:
            Speaker 1 (00:00:00):
            text here
        remove_fillers: Whether to remove Estonian filler words

    Returns:
        List of segment dictionaries with speaker and text fields.
    """
    segments = []

    # Pattern to match: Speaker N (HH:MM:SS):
    pattern = r"^Speaker (\d+) \((\d{2}:\d{2}:\d{2})\):\s*$"

    lines = content.strip().split("\n")
    current_speaker = None
    current_text = []
    current_timestamp = None

    for line in lines:
        match = re.match(pattern, line.strip())

        if match:
            # Save previous segment if exists
            if current_speaker and current_text:
                text = " ".join(current_text).strip()
                if remove_fillers:
                    text = clean_fillers(text)

                # Skip empty segments after filler removal
                if text:
                    segments.append(
                        {
                            "speaker": f"SPEAKER_{current_speaker:02d}",
                            "timestamp": current_timestamp,
                            "text": text,
                        }
                    )
                current_text = []

            # Start new segment
            current_speaker = int(match.group(1))
            current_timestamp = match.group(2)
        else:
            # Accumulate text for current speaker
            text = line.strip()
            if text:
                current_text.append(text)

    # Don't forget the last segment
    if current_speaker and current_text:
        text = " ".join(current_text).strip()
        if remove_fillers:
            text = clean_fillers(text)

        # Skip empty segments after filler removal
        if text:
            segments.append(
                {
                    "speaker": f"SPEAKER_{current_speaker:02d}",
                    "timestamp": current_timestamp,
                    "text": text,
                }
            )

    return segments


@app.command()
def convert(
    input_file: Path = typer.Option(
        ...,
        "--input",
        "-i",
        help="Input transcript text file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    output_file: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Output JSON file",
    ),
    remove_fillers: bool = typer.Option(
        False,
        "--remove-fillers",
        "-f",
        help="Remove Estonian filler words (mhm, aa, vä, nagu, etc.)",
    ),
) -> None:
    """Convert plain text transcript to JSON format for ASR evaluation."""

    typer.echo(f"Reading transcript from: {input_file}")

    # Read the input file
    with open(input_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Parse the transcript
    segments = parse_transcript(content, remove_fillers=remove_fillers)

    if remove_fillers:
        typer.echo("Removed filler words from transcript")
    typer.echo(f"Parsed {len(segments)} segments")

    # Create output structure
    output_data = {"segments": segments}

    # Write JSON output
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    typer.echo(f"Saved JSON to: {output_file}")

    # Show preview
    typer.echo("\nFirst segment preview:")
    if segments:
        typer.echo(json.dumps(segments[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    app()
