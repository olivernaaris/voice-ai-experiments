#!/usr/bin/env uv run
"""ASR Evaluation Tool for Whisper Diarization.

This module provides comprehensive evaluation capabilities for automatic speech
recognition (ASR) systems by comparing reference (ground truth) transcriptions
against hypothesis (model output) transcriptions.

Features:
    - Multiple evaluation metrics (WER, MER, WIL, WIP, CER)
    - Word-level alignment visualization
    - Per-speaker analysis for diarized transcriptions
    - Detailed error categorization (substitutions, deletions, insertions)

The tool expects JSON files with the following structure:
    {
        "segments": [
            {
                "speaker": "SPEAKER_00",
                "text": "Transcribed text here",
                ...
            }
        ]
    }

Usage:
    ./test_asr_evaluation.py \
        --reference-file path/to/reference.json \
        --hypothesis-file path/to/hypothesis.json

Metrics Explained:
    - WER (Word Error Rate): Percentage of words that were incorrectly predicted
    - MER (Match Error Rate): Normalized edit distance at word level
    - WIL (Word Information Lost): Measure of information loss in the hypothesis
    - WIP (Word Information Preserved): Measure of information retained (1 - WIL)
    - CER (Character Error Rate): Similar to WER but at character level
"""

import json
from pathlib import Path

import typer
from jiwer import wer, mer, wil, wip, cer
from jiwer import process_words, visualize_alignment

# Custom imports
from utils.logging import logger

app = typer.Typer(
    help="ASR Evaluation Tool - Compare reference and hypothesis transcriptions"
)


def load_json_file(file_path: Path) -> dict:
    """Load and parse a JSON file containing transcription segments.

    Args:
        file_path: Path to the JSON file to load.

    Returns:
        Dictionary containing the parsed JSON data with 'segments' key.

    Raises:
        json.JSONDecodeError: If the file contains invalid JSON.
        FileNotFoundError: If the file does not exist.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_text_from_segments(segments: list[dict]) -> str:
    """Extract and concatenate text from transcription segments.

    Processes a list of segment dictionaries and combines their text content
    into a single string, with segments separated by spaces.

    Args:
        segments: List of segment dictionaries, each containing a 'text' key.

    Returns:
        Single string with all segment texts concatenated and whitespace normalized.

    Example:
        >>> segments = [
        ...     {"text": " Hello ", "speaker": "SPEAKER_00"},
        ...     {"text": "world ", "speaker": "SPEAKER_01"}
        ... ]
        >>> extract_text_from_segments(segments)
        'Hello world'
    """
    return " ".join([segment["text"].strip() for segment in segments])


def compute_metrics(reference_text: str, hypothesis_text: str) -> None:
    """Compute and log comprehensive ASR evaluation metrics.

    Calculates multiple error rates and information metrics to provide
    a complete picture of ASR system performance. All metrics are logged
    to the console with proper formatting.

    Args:
        reference_text: Ground truth transcription text.
        hypothesis_text: ASR system output text to evaluate.

    Metrics Computed:
        - WER: Word Error Rate - primary metric for ASR quality
        - MER: Match Error Rate - normalized edit distance
        - WIL: Word Information Lost - information theory metric
        - WIP: Word Information Preserved - complement of WIL
        - CER: Character Error Rate - fine-grained error analysis
    """
    logger.info("=" * 80)
    logger.info("EVALUATION METRICS:")
    logger.info("=" * 80)
    logger.info(
        f"Word Error Rate (WER):           {wer(reference_text, hypothesis_text):.2%}"
    )
    logger.info(
        f"Match Error Rate (MER):          {mer(reference_text, hypothesis_text):.2%}"
    )
    logger.info(
        f"Word Information Lost (WIL):     {wil(reference_text, hypothesis_text):.2%}"
    )
    logger.info(
        f"Word Information Preserved (WIP): {wip(reference_text, hypothesis_text):.2%}"
    )
    logger.info(
        f"Character Error Rate (CER):      {cer(reference_text, hypothesis_text):.2%}"
    )


def show_alignment(reference_text: str, hypothesis_text: str) -> None:
    """Show detailed word-level alignment between reference and hypothesis.

    Performs alignment analysis and visualizes how words in the hypothesis
    correspond to words in the reference transcription. Categorizes errors
    into substitutions, deletions, and insertions.

    Args:
        reference_text: Ground truth transcription text.
        hypothesis_text: ASR system output text to evaluate.

    Output includes:
        - Count of substitutions (wrong word predicted)
        - Count of deletions (word missed in hypothesis)
        - Count of insertions (extra word in hypothesis)
        - Count of hits (correctly predicted words)
        - Visual alignment showing matched and mismatched words
    """
    alignment = process_words(reference_text, hypothesis_text)
    logger.info(f"\nSubstitutions: {alignment.substitutions}")
    logger.info(f"Deletions:     {alignment.deletions}")
    logger.info(f"Insertions:    {alignment.insertions}")
    logger.info(f"Hits:          {alignment.hits}")

    logger.info("\n" + "=" * 80)
    logger.info("WORD-LEVEL ALIGNMENT:")
    logger.info("=" * 80)
    logger.info(visualize_alignment(alignment))


def analyze_speakers(reference_data: dict, hypothesis_data: dict) -> None:
    """Perform per-speaker diarization and transcription accuracy analysis.

    Aggregates segments by speaker and computes Word Error Rate for each
    individual speaker. This helps identify if the ASR system performs
    differently for different speakers, which is crucial for evaluating
    diarization quality.

    Args:
        reference_data: Dictionary containing reference segments with speaker labels.
        hypothesis_data: Dictionary containing hypothesis segments with speaker labels.

    Note:
        Only speakers present in both reference and hypothesis with non-empty
        text are included in the analysis. Speaker labels should match between
        reference and hypothesis for meaningful comparison.
    """
    logger.info("\n" + "=" * 80)
    logger.info("PER-SPEAKER ANALYSIS:")
    logger.info("=" * 80)

    ref_speakers = {}
    hyp_speakers = {}

    for seg in reference_data["segments"]:
        speaker = seg["speaker"]
        ref_speakers.setdefault(speaker, []).append(seg["text"].strip())

    for seg in hypothesis_data["segments"]:
        speaker = seg["speaker"]
        hyp_speakers.setdefault(speaker, []).append(seg["text"].strip())

    all_speakers = set(ref_speakers.keys()) | set(hyp_speakers.keys())
    for speaker in sorted(all_speakers):
        ref_text = " ".join(ref_speakers.get(speaker, []))
        hyp_text = " ".join(hyp_speakers.get(speaker, []))

        if ref_text and hyp_text:
            logger.info(f"{speaker}: WER = {wer(ref_text, hyp_text):.2%}")


@app.command()
def evaluate(
    reference_file: Path = typer.Option(
        ...,
        "--reference-file",
        "-r",
        help="Path to reference JSON file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    hypothesis_file: Path = typer.Option(
        ...,
        "--hypothesis-file",
        "-h",
        help="Path to hypothesis JSON file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
) -> None:
    """Evaluate ASR output by comparing reference and hypothesis transcriptions.

    This command orchestrates a complete evaluation workflow:
    1. Loads reference (ground truth) and hypothesis (ASR output) JSON files
    2. Extracts text from segments in both files
    3. Computes standard ASR evaluation metrics
    4. Shows detailed word-level alignment visualization
    5. Performs per-speaker accuracy analysis

    Args:
        reference_file: Path to JSON file with ground truth transcriptions.
        hypothesis_file: Path to JSON file with ASR system output.

    Example:
        ./test_asr_evaluation.py \
            -r data/reference.json \
            -h output/hypothesis.json
    """
    # Load the JSON files
    logger.info(f"Loading reference file: {reference_file}")
    reference_data = load_json_file(reference_file)

    logger.info(f"Loading hypothesis file: {hypothesis_file}")
    hypothesis_data = load_json_file(hypothesis_file)

    # Extract text from segments
    reference_text = extract_text_from_segments(reference_data["segments"])
    hypothesis_text = extract_text_from_segments(hypothesis_data["segments"])

    # Compute metrics
    compute_metrics(reference_text, hypothesis_text)

    # Show alignment
    show_alignment(reference_text, hypothesis_text)

    # Analyze speakers
    analyze_speakers(reference_data, hypothesis_data)


if __name__ == "__main__":
    app()
