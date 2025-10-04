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


def parse_transcript(content: str) -> list[dict]:
    """Parse transcript with speaker labels and timestamps into segments.

    Args:
        content: Raw transcript text with format like:
            Speaker 1 (00:00:00):
            text here

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
                segments.append(
                    {
                        "speaker": f"SPEAKER_{current_speaker:02d}",
                        "timestamp": current_timestamp,
                        "text": " ".join(current_text).strip(),
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
        segments.append(
            {
                "speaker": f"SPEAKER_{current_speaker:02d}",
                "timestamp": current_timestamp,
                "text": " ".join(current_text).strip(),
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
) -> None:
    """Convert plain text transcript to JSON format for ASR evaluation."""

    typer.echo(f"Reading transcript from: {input_file}")

    # Read the input file
    with open(input_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Parse the transcript
    segments = parse_transcript(content)

    typer.echo(f"Parsed {len(segments)} segments")

    # Create output structure
    output_data = {"segments": segments}

    # Write JSON output
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    typer.echo(f"✓ Saved JSON to: {output_file}")

    # Show preview
    typer.echo("\nFirst segment preview:")
    if segments:
        typer.echo(json.dumps(segments[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    app()
