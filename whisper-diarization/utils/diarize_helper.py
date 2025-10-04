"""Helper functions for audio diarization with speaker ordering."""

import torchaudio
from pyannote.audio.pipelines.utils.hook import ProgressHook

# Custom imports
from .logging import logger


def build_appearance_mapping(annotation) -> dict[str, str]:
    """
    Build speaker mapping based on chronological first appearance.

    This function creates a mapping from pyannote's arbitrary speaker labels
    (e.g., SPEAKER_00, SPEAKER_01) to sequential labels based on when each
    speaker first appears in the audio timeline (SPEAKER_1, SPEAKER_2, etc.).

    Args:
        annotation: pyannote.core.Annotation object containing speaker segments

    Returns:
        dict: Mapping from original speaker labels to appearance-ordered labels
              Example: {"SPEAKER_00": "SPEAKER_1", "SPEAKER_01": "SPEAKER_2"}
    """
    speaker_map = {}
    speaker_counter = 1

    # Iterate chronologically through timeline
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        if speaker not in speaker_map:
            speaker_map[speaker] = f"SPEAKER_{speaker_counter}"
            speaker_counter += 1

    return speaker_map


def diarize_audio(
    diarization_model,
    audio_file_wav: str,
    num_speakers: int | None = None,
) -> tuple[list[dict], int]:
    """
    Perform speaker diarization using exclusive mode for better STT alignment.

    This function uses pyannote's exclusive_speaker_diarization mode, which
    assigns only one speaker at a time (the dominant speaker). This matches
    Whisper's behavior of transcribing the dominant speaker and simplifies
    speaker assignment logic.

    Speakers are automatically numbered by order of first appearance
    (SPEAKER_1, SPEAKER_2, etc.) rather than pyannote's arbitrary clustering order.

    Args:
        diarization_model: Pyannote diarization pipeline model
        audio_file_wav: Path to the audio file in WAV format
        num_speakers: Optional number of speakers to detect. If None, model will
                     automatically determine the number of speakers.

    Returns:
        tuple containing:
            - diarize_segments: List of dicts with keys "start", "end", "speaker"
            - detected_num_speakers: Number of unique speakers detected

    Example:
        >>> segments, num_speakers = diarize_audio(model, "audio.wav", num_speakers=2)
        >>> print(segments[0])
        {"start": 0.5, "end": 3.2, "speaker": "SPEAKER_1"}
    """
    waveform, sample_rate = torchaudio.load(audio_file_wav)

    with ProgressHook() as hook:
        diarization = diarization_model(
            {"waveform": waveform, "sample_rate": sample_rate},
            num_speakers=num_speakers,
            hook=hook,
        )

    # Use exclusive_speaker_diarization for cleaner STT alignment
    # This mode ensures only one speaker is active at a time (the dominant speaker)
    annotation = diarization.exclusive_speaker_diarization

    # Relabel speakers by chronological appearance
    speaker_map = build_appearance_mapping(annotation)
    if speaker_map:
        annotation = annotation.rename_labels(mapping=speaker_map)
        logger.info(f"Relabeled speakers by appearance: {speaker_map}")

    # Convert to segments list
    diarize_segments = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        diarize_segments.append(
            {"start": turn.start, "end": turn.end, "speaker": speaker}
        )

    unique_speakers = {
        speaker for _, _, speaker in annotation.itertracks(yield_label=True)
    }
    detected_num_speakers = len(unique_speakers)

    return diarize_segments, detected_num_speakers
