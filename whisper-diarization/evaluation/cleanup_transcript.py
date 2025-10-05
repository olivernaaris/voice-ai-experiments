#!/usr/bin/env uv run
"""Clean up whisper-diarization JSON output to a simpler format.

This script transforms verbose whisper-diarization output into a simplified
format suitable for evaluation and analysis.
"""

import orjson
from pathlib import Path

import typer

app = typer.Typer(help="Clean up whisper-diarization JSON output")


def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted timestamp string (HH:MM:SS)
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def cleanup_transcript(input_file: Path, output_file: Path) -> list[dict[str, str]]:
    """Clean up the transcript JSON to a simpler format.

    Args:
        input_file: Path to input JSON file with verbose diarization data
        output_file: Path to output JSON file for simplified data

    Returns:
        List of simplified segment dictionaries
    """
    with open(input_file, "rb") as f:
        data = orjson.loads(f.read())

    simplified_segments = []
    for segment in data.get("segments", []):
        simplified_segment = {
            "speaker": segment["speaker"],
            "timestamp": format_timestamp(segment["start"]),
            "text": segment["text"],
        }
        simplified_segments.append(simplified_segment)

    result = {"segments": simplified_segments}

    with open(output_file, "wb") as f:
        f.write(orjson.dumps(result, option=orjson.OPT_INDENT_2))

    return simplified_segments


@app.command()
def convert(
    input_file: Path = typer.Option(
        ...,
        "--input",
        "-i",
        help="Input JSON file with verbose diarization data",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    output_file: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Output JSON file for simplified data",
    ),
) -> None:
    """Clean up whisper-diarization JSON output to a simpler format."""

    typer.echo(f"Reading transcript from: {input_file}")

    segments = cleanup_transcript(input_file, output_file)

    typer.echo(f"Cleaned up transcript saved to: {output_file}")
    typer.echo(f"Total segments: {len(segments)}")

    # Show preview
    typer.echo("\nFirst segment preview:")
    if segments:
        typer.echo(orjson.dumps(segments[0], option=orjson.OPT_INDENT_2).decode())


if __name__ == "__main__":
    app()
