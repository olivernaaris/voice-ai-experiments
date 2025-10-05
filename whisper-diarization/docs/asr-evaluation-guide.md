# ASR Evaluation Guide

This guide explains how to use the evaluation tools to calculate WER (Word Error Rate) and other metrics for your transcripts.

## Overview

The evaluation process requires two JSON files:
1. **Reference (ground truth)**: The correct transcription
2. **Hypothesis (ASR output)**: The transcription to evaluate

## Step-by-Step Process

### Step 1: Convert Your Transcript to JSON Format

If you have a plain text transcript with speaker labels, first convert it to JSON:

```bash
./evaluation/convert_text_transcript_to_json.py \
    --input evaluation/input-2min-wav-et.txt \
    --output evaluation/reference-2min-wav-et.json
```

This will create a JSON file with the following structure:
```json
{
  "segments": [
    {
      "speaker": "SPEAKER_01",
      "timestamp": "00:00:00",
      "text": "käisime vaatasime ühte korterit..."
    }
  ]
}
```

### Step 2: Cleanup your Hypothesis File JSON

./cleanup_transcript.py \
 --input ../whisper-large-v3-turbo-et-verbatim-ct2_20251004_141446.json \
 --output evaluation/cleaned_output_whisper-large-v3-turbo-et-verbatim-ct2_20251004_141446.json

You need a hypothesis file (the ASR system output to evaluate). This should be in the same JSON format:

```json
{
  "segments": [
    {
      "speaker": "SPEAKER_01",
      "text": "käisime vaatasime ühte korterit..."
    }
  ]
}
```

If you have output from the whisper-diarization pipeline, it should already be in the correct format.

### Step 3: Run the Evaluation

Once you have both files, run the evaluation:

```bash
./evaluation/test_asr_evaluation.py \
    --reference-file ./evaluation/reference-2min-wav-et.json \
    --hypothesis-file ./evaluation/hypothesis_whisper-large-v3-turbo-et-verbatim-ct2_20251004_141446.json
```
```

## Output Metrics Explained

The evaluation will provide several metrics:

### Word Error Rate (WER)
- **Primary metric** for ASR quality
- Percentage of words incorrectly predicted
- Lower is better (0% = perfect)
- Formula: `(Substitutions + Deletions + Insertions) / Total Reference Words`

### Match Error Rate (MER)
- Normalized edit distance at word level
- Similar to WER but treats consecutive errors differently
- Lower is better

### Word Information Lost (WIL)
- Information theory metric
- Measures information loss in hypothesis
- Range: 0% (no loss) to 100% (complete loss)

### Word Information Preserved (WIP)
- Complement of WIL: `WIP = 1 - WIL`
- Higher is better (100% = perfect)

### Character Error Rate (CER)
- Similar to WER but at character level
- Useful for languages with compound words or agglutination
- Lower is better

### Per-Speaker Analysis
- Individual WER for each speaker
- Helps identify if ASR performs differently for different speakers
- Useful for evaluating diarization quality

## Example Output

```
================================================================================
EVALUATION METRICS:
================================================================================
Word Error Rate (WER):           15.32%
Match Error Rate (MER):          12.45%
Word Information Lost (WIL):     18.76%
Word Information Preserved (WIP): 81.24%
Character Error Rate (CER):      8.92%

Substitutions: 12
Deletions:     5
Insertions:    3
Hits:          180

================================================================================
WORD-LEVEL ALIGNMENT:
================================================================================
REF: käisime vaatasime ühte korterit
HYP: käisime vaatasime ühte ******
        S           S           S    D

================================================================================
PER-SPEAKER ANALYSIS:
================================================================================
SPEAKER_01: WER = 12.50%
SPEAKER_02: WER = 16.75%
SPEAKER_03: WER = 18.20%
```

Legend for alignment:
- `S` = Substitution (wrong word)
- `D` = Deletion (word missing)
- `I` = Insertion (extra word)
- `******` = Deleted word marker

## File Format Requirements

Both reference and hypothesis JSON files must have:
- A `segments` array at the root level
- Each segment must have a `text` field
- Optional: `speaker` field for per-speaker analysis
- Optional: `timestamp` or other metadata fields (ignored by evaluation)

## Tips for Best Results

1. **Text Normalization**: Ensure both reference and hypothesis use consistent:
   - Punctuation (or lack thereof)
   - Case (uppercase/lowercase)
   - Number formatting (digits vs. words)

2. **Speaker Labels**: For per-speaker analysis, ensure speaker labels match between reference and hypothesis

3. **Language Consistency**: Both files should use the same language and encoding (UTF-8)

4. **Segment Granularity**: Smaller segments provide more detailed error localization

## Troubleshooting

### "Segments not found" error
- Check that your JSON has a top-level `segments` key
- Verify the JSON is valid (use `python -m json.tool yourfile.json`)

### High WER despite good transcription
- Check text normalization (case, punctuation, spacing)
- Verify you're comparing the same audio segments
- Check speaker label alignment

### Import errors
- Ensure you're in the correct directory: `cd whisper-diarization`
- Verify dependencies are installed: `uv sync`
