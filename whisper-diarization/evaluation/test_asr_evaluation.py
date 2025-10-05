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

import jiwer
import typer
from jiwer import wer, mer, wil, wip, cer
from jiwer import process_words, visualize_alignment, Compose

# Custom imports
from utils.logging import logger

app = typer.Typer(
    help="ASR Evaluation Tool - Compare reference and hypothesis transcriptions"
)

transforms = Compose(
    [
        jiwer.ExpandCommonEnglishContractions(),
        jiwer.RemoveEmptyStrings(),
        jiwer.ToLowerCase(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.RemovePunctuation(),
        jiwer.ReduceToSingleSentence(),
        jiwer.ReduceToListOfListOfWords(),
    ]
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


def extract_text_from_segments(segments: list[dict]) -> list[str]:
    """Extract text from each segment as separate sentences.

    Returns a list of individual segment texts instead of concatenating them.
    This allows jiwer to compare segments one-by-one for better alignment.

    Args:
        segments: List of segment dictionaries, each containing a 'text' key.

    Returns:
        List of strings, one per segment.

    Example:
        >>> segments = [
        ...     {"text": " Hello ", "speaker": "SPEAKER_00"},
        ...     {"text": "world ", "speaker": "SPEAKER_01"}
        ... ]
        >>> extract_text_from_segments(segments)
        ['Hello', 'world']
    """
    return [segment["text"].strip() for segment in segments]


def align_segments_by_timing(
    reference_segments: list[dict],
    hypothesis_segments: list[dict],
    tolerance: float = 1.0,
) -> tuple[list[str], list[str]]:
    """Align segments by timing to handle different numbers of segments.

    Uses a simple greedy algorithm to match hypothesis segments to reference
    segments based on temporal overlap. This handles cases where the ASR system
    produces different segmentation than the reference annotation.

    Args:
        reference_segments: Reference segments with timing information
        hypothesis_segments: Hypothesis segments with timing information
        tolerance: Maximum time gap allowed for matching segments (in seconds)

    Returns:
        Tuple of (aligned_reference_texts, aligned_hypothesis_texts) with
        matching lengths for jiwer evaluation.
    """
    ref_texts = []
    hyp_texts = []

    ref_idx = 0
    hyp_idx = 0

    while ref_idx < len(reference_segments) and hyp_idx < len(hypothesis_segments):
        ref_seg = reference_segments[ref_idx]
        hyp_seg = hypothesis_segments[hyp_idx]

        # Check for temporal overlap
        ref_start = ref_seg.get("start", 0)
        ref_end = ref_seg.get("end", 0)
        hyp_start = hyp_seg.get("start", 0)
        hyp_end = hyp_seg.get("end", 0)

        # Check if segments overlap significantly
        overlap_start = max(ref_start, hyp_start)
        overlap_end = min(ref_end, hyp_end)

        if overlap_end - overlap_start > 0:
            # Segments overlap, include them in evaluation
            ref_texts.append(ref_seg["text"].strip())
            hyp_texts.append(hyp_seg["text"].strip())
            ref_idx += 1
            hyp_idx += 1
        elif hyp_end < ref_start - tolerance:
            # Hypothesis segment is before reference segment
            hyp_idx += 1
        elif ref_end < hyp_start - tolerance:
            # Reference segment is before hypothesis segment
            ref_idx += 1
        else:
            # No clear temporal relationship, advance the one that ends first
            if ref_end < hyp_end:
                ref_idx += 1
            else:
                hyp_idx += 1

    return ref_texts, hyp_texts


def compute_metrics(reference_text: list, hypothesis_text: list) -> None:
    """Compute and log comprehensive ASR evaluation metrics.

    Calculates multiple error rates and information metrics to provide
    a complete picture of ASR system performance. All metrics are logged
    to the console with proper formatting.

    Text normalization is applied using the global transforms before computing
    metrics, which includes: expanding contractions, lowercasing, removing
    punctuation, and normalizing whitespace.

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
    logger.info("EVALUATION METRICS (with text normalization):")
    logger.info("=" * 80)
    logger.info(
        f"Word Error Rate (WER):           {wer(reference_text, hypothesis_text, reference_transform=transforms, hypothesis_transform=transforms):.2%}"
    )
    logger.info(
        f"Match Error Rate (MER):          {mer(reference_text, hypothesis_text, reference_transform=transforms, hypothesis_transform=transforms):.2%}"
    )
    logger.info(
        f"Word Information Lost (WIL):     {wil(reference_text, hypothesis_text, reference_transform=transforms, hypothesis_transform=transforms):.2%}"
    )
    logger.info(
        f"Word Information Preserved (WIP): {wip(reference_text, hypothesis_text, reference_transform=transforms, hypothesis_transform=transforms):.2%}"
    )
    logger.info(
        f"Character Error Rate (CER):      {cer(reference_text, hypothesis_text, reference_transform=transforms, hypothesis_transform=transforms):.2%}"
    )


def show_alignment(reference_text: list, hypothesis_text: list) -> None:
    """Show detailed word-level alignment between reference and hypothesis.

    Performs alignment analysis and visualizes how words in the hypothesis
    correspond to words in the reference transcription. Categorizes errors
    into substitutions, deletions, and insertions.

    Text normalization is applied before alignment for fair comparison.

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
    alignment = process_words(
        reference_text,
        hypothesis_text,
        reference_transform=transforms,
        hypothesis_transform=transforms,
    )
    logger.info(f"\nSubstitutions: {alignment.substitutions}")
    logger.info(f"Deletions:     {alignment.deletions}")
    logger.info(f"Insertions:    {alignment.insertions}")
    logger.info(f"Hits:          {alignment.hits}")

    logger.info("\n" + "=" * 80)
    logger.info("WORD-LEVEL ALIGNMENT (normalized text):")
    logger.info("=" * 80)
    logger.info(visualize_alignment(alignment))


def analyze_speakers(reference_data: dict, hypothesis_data: dict) -> None:
    """Perform per-speaker diarization and transcription accuracy analysis.

    Aggregates segments by speaker and computes Word Error Rate for each
    individual speaker. This helps identify if the ASR system performs
    differently for different speakers, which is crucial for evaluating
    diarization quality.

    Text normalization is applied per speaker for consistent comparison.

    Args:
        reference_data: Dictionary containing reference segments with speaker labels.
        hypothesis_data: Dictionary containing hypothesis segments with speaker labels.

    Note:
        Only speakers present in both reference and hypothesis with non-empty
        text are included in the analysis. Speaker labels should match between
        reference and hypothesis for meaningful comparison.
    """
    logger.info("\n" + "=" * 80)
    logger.info("PER-SPEAKER ANALYSIS (with text normalization):")
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
            logger.info(
                f"{speaker}: WER = {wer(ref_text, hyp_text, reference_transform=transforms, hypothesis_transform=transforms):.2%}"
            )


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
