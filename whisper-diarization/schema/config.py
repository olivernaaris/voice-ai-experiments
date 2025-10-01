"""Configuration schemas for Whisper diarization pipeline"""

from pathlib import Path

from pydantic import Field

from .base import CustomBaseModel
from .cli import CLIArgs


class FileInputConfig(CustomBaseModel):
    """Configuration for audio file input sources."""

    file_string: str | None = Field(
        None, description="File content as a base64 encoded string."
    )
    file_url: str | None = Field(None, description="URL to an audio file.")
    file_path: Path | None = Field(None, description="Path to an audio file.")

    def get_input_source(self) -> str | None:
        """Get the first available input source."""
        if self.file_string:
            return self.file_string
        elif self.file_url:
            return self.file_url
        elif self.file_path:
            return str(self.file_path)
        return None


class ModelConfig(CustomBaseModel):
    """Configuration for the Whisper model."""

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


class TranscriptionConfig(CustomBaseModel):
    """Configuration for transcription parameters."""

    num_speakers: int | None = Field(None, description="Number of speakers.", ge=1)
    translate: bool = Field(False, description="Translate the audio to English.")
    language: str | None = Field(
        None, description="Language of the audio.", min_length=2, max_length=5
    )
    prompt: str | None = Field(None, description="Initial prompt for the model.")


class PreprocessingConfig(CustomBaseModel):
    """Configuration for audio preprocessing."""

    preprocess: int = Field(0, description="Preprocessing level (0-4).", ge=0, le=4)
    highpass_freq: int = Field(45, description="Highpass filter frequency.", gt=0)
    lowpass_freq: int = Field(8000, description="Lowpass filter frequency.", gt=0)
    prop_decrease: float = Field(
        0.3, description="Proportion to decrease noise.", ge=0.0, le=1.0
    )
    stationary: bool = Field(True, description="Whether the noise is stationary.")
    target_dBFS: float = Field(-18.0, description="Target dBFS for normalization.")


class OutputConfig(CustomBaseModel):
    """Configuration for output settings."""

    output_filename: str | None = Field(
        None, description="Base name for the output JSON file."
    )


class WhisperDiarizationConfig(CustomBaseModel):
    """Main configuration for Whisper diarization pipeline.

    This class consolidates all configuration options into a single,
    type-safe model that can be used throughout the application.
    """

    # File input configuration
    file_input: FileInputConfig = Field(
        default_factory=FileInputConfig, description="Audio file input configuration."
    )

    # Model configuration
    model: ModelConfig = Field(
        default_factory=ModelConfig, description="Whisper model configuration."
    )

    # Transcription configuration
    transcription: TranscriptionConfig = Field(
        default_factory=TranscriptionConfig, description="Transcription parameters."
    )

    # Preprocessing configuration
    preprocessing: PreprocessingConfig = Field(
        default_factory=PreprocessingConfig,
        description="Audio preprocessing configuration.",
    )

    # Output configuration
    output: OutputConfig = Field(
        default_factory=OutputConfig, description="Output configuration."
    )

    def model_post_init(self, __context) -> None:
        """Validate configuration after initialization."""
        # Ensure at least one input source is provided
        if not self.file_input.get_input_source():
            raise ValueError(
                "At least one input source must be provided: "
                "file_string, file_url, or file_path"
            )

        # Validate preprocessing level compatibility
        if self.preprocessing.preprocess > 0:
            if self.preprocessing.highpass_freq >= self.preprocessing.lowpass_freq:
                raise ValueError(
                    "Highpass frequency must be less than lowpass frequency"
                )

    @classmethod
    def from_cli_args(cls, cli_args: CLIArgs) -> "WhisperDiarizationConfig":
        """Create configuration from CLI arguments model.

        Args:
            cli_args: Type-safe CLI arguments model containing all configuration options.

        Returns:
            WhisperDiarizationConfig: Fully configured diarization configuration.
        """
        return cls(
            file_input=FileInputConfig(
                file_string=cli_args.file_string,
                file_url=cli_args.file_url,
                file_path=cli_args.file_path,
            ),
            model=ModelConfig(
                device=cli_args.device,
                compute_type=cli_args.compute_type,
                model_name=cli_args.model_name,
            ),
            transcription=TranscriptionConfig(
                num_speakers=cli_args.num_speakers,
                translate=cli_args.translate,
                language=cli_args.language,
                prompt=cli_args.prompt,
            ),
            preprocessing=PreprocessingConfig(
                preprocess=cli_args.preprocess,
                highpass_freq=cli_args.highpass_freq,
                lowpass_freq=cli_args.lowpass_freq,
                prop_decrease=cli_args.prop_decrease,
                stationary=cli_args.stationary,
                target_dBFS=cli_args.target_dBFS,
            ),
            output=OutputConfig(
                output_filename=cli_args.output_filename,
            ),
        )
