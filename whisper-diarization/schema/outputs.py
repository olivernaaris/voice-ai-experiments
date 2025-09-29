"""Data structures for pipeline outputs."""

from __future__ import annotations


class Output:
    """Data structure for diarization pipeline results.

    This class encapsulates the output of the WhisperDiarizationPipeline,
    containing the transcribed segments with speaker information, detected
    language, and number of speakers.
    """

    def __init__(
        self,
        segments: list[dict],
        language: str | None = None,
        num_speakers: int | None = None,
    ) -> None:
        """Initialize the output data structure.

        Args:
            segments: List of transcribed segments with speaker information
            language: Detected language code (e.g., 'en', 'es')
            num_speakers: Number of speakers detected in the audio
        """
        self.segments = segments
        self.language = language
        self.num_speakers = num_speakers

    def to_dict(self) -> dict:
        """Convert the output to a dictionary format.

        Returns:
            Dictionary containing segments, language, and num_speakers
        """
        return {
            "segments": self.segments,
            "language": self.language,
            "num_speakers": self.num_speakers,
        }
