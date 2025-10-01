from __future__ import annotations

from pydantic import BaseModel, Field
from typing import List, Optional


class Word(BaseModel):
    """A word with a start and end time, and a probability."""

    start: float = Field(..., description="The start time of the word in seconds.")
    end: float = Field(..., description="The end time of the word in seconds.")
    word: str = Field(..., description="The word that was spoken.")
    probability: float = Field(..., description="The probability of the word.")
    speaker: Optional[str] = Field(None, description="The speaker of the word.")


class Segment(BaseModel):
    """A segment of speech with a start and end time, and a list of words."""

    start: float = Field(..., description="The start time of the segment in seconds.")
    end: float = Field(..., description="The end time of the segment in seconds.")
    text: str = Field(..., description="The text of the segment.")
    speaker: Optional[str] = Field(None, description="The speaker of the segment.")
    avg_logprob: Optional[float] = Field(
        None, description="The average log probability of the segment."
    )
    words: List[Word] = Field(..., description="The words in the segment.")
    duration: Optional[float] = Field(
        None, description="The duration of the segment in seconds."
    )


class DiarizationEntry(BaseModel):
    """A single entry from the diarization model."""

    start: float = Field(..., description="The start time of the entry in seconds.")
    end: float = Field(..., description="The end time of the entry in seconds.")
    speaker: str = Field(..., description="The speaker of the entry.")
