from pydantic import Field

from .base import CustomBaseModel


class Word(CustomBaseModel):
    """A word with a start and end time, and a probability."""

    start: float = Field(..., description="The start time of the word in seconds.")
    end: float = Field(..., description="The end time of the word in seconds.")
    word: str = Field(..., description="The word that was spoken.")
    probability: float = Field(..., description="The probability of the word.")
    speaker: str | None = Field(None, description="The speaker of the word.")


class Segment(CustomBaseModel):
    """A segment of speech with a start and end time, and a list of words."""

    start: float = Field(..., description="The start time of the segment in seconds.")
    end: float = Field(..., description="The end time of the segment in seconds.")
    text: str = Field(..., description="The text of the segment.")
    speaker: str | None = Field(None, description="The speaker of the segment.")
    avg_logprob: float | None = Field(
        None, description="The average log probability of the segment."
    )
    words: list[Word] = Field(..., description="The words in the segment.")
    duration: float | None = Field(
        None, description="The duration of the segment in seconds."
    )


class DiarizationEntry(CustomBaseModel):
    """A single entry from the diarization model."""

    start: float = Field(..., description="The start time of the entry in seconds.")
    end: float = Field(..., description="The end time of the entry in seconds.")
    speaker: str = Field(..., description="The speaker of the entry.")


class Output(CustomBaseModel):
    """Data structure for diarization pipeline results.

    This class encapsulates the output of the WhisperDiarizationPipeline,
    containing the transcribed segments with speaker information, detected
    language, and number of speakers.
    """

    segments: list[Segment]
    language: str | None = None
    num_speakers: int | None = None
