"""CLI argument schemas for Whisper diarization pipeline"""

from pathlib import Path

from pydantic import Field

from .base import CustomBaseModel


class CLIArgs(CustomBaseModel):
    """CLI arguments model for type-safe argument handling.

    This model consolidates all CLI arguments into a single type-safe
    pydantic model that can be used throughout the application.
    """

    # File input configuration
    file_string: str | None = Field(
        None, description="File content as a base64 encoded string."
    )
    file_url: str | None = Field(None, description="URL to an audio file.")
    file_path: Path | None = Field(None, description="Path to an audio file.")

    # Transcription configuration
    num_speakers: int | None = Field(None, description="Number of speakers.", ge=1)
    translate: bool = Field(False, description="Translate the audio to English.")
    language: str | None = Field(
        None, description="Language of the audio.", min_length=2, max_length=5
    )
    prompt: str | None = Field(None, description="Initial prompt for the model.")

    # Preprocessing configuration
    preprocess: int = Field(0, description="Preprocessing level (0-4).", ge=0, le=4)
    highpass_freq: int = Field(45, description="Highpass filter frequency.", gt=0)
    lowpass_freq: int = Field(8000, description="Lowpass filter frequency.", gt=0)
    prop_decrease: float = Field(
        0.3, description="Proportion to decrease noise.", ge=0.0, le=1.0
    )
    stationary: bool = Field(True, description="Whether the noise is stationary.")
    target_dBFS: float = Field(-18.0, description="Target dBFS for normalization.")

    # Model configuration
    device: str = Field(
        "cpu",
        description="Device to use for computation (cpu or cuda).",
        pattern="^(cpu|cuda)$",
    )
    compute_type: str = Field("int8", description="Compute type for the model.")
    model_name: str = Field(
        "whisper-large-v3-turbo-et-verbatim-ct2",
        description="Name of the Whisper model to use.",
    )

    # Output configuration
    output_filename: str | None = Field(
        None, description="Base name for the output JSON file."
    )

    def model_post_init(self, __context) -> None:
        """Validate CLI arguments after initialization."""
        # Ensure at least one input source is provided
        if not any([self.file_string, self.file_url, self.file_path]):
            raise ValueError(
                "At least one input source must be provided: "
                "file_string, file_url, or file_path"
            )

        # Validate preprocessing level compatibility
        if self.preprocess > 0:
            if self.highpass_freq >= self.lowpass_freq:
                raise ValueError(
                    "Highpass frequency must be less than lowpass frequency"
                )
