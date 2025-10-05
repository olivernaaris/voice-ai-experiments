#!/usr/bin/env uv run

from pathlib import Path
from typing import Annotated, Optional

import typer

# Custom imports
from pipeline import WhisperDiarizationPipeline
from utils import logger, write_json_file

# Custom schemas
from schema.config import WhisperDiarizationConfig
from schema.cli import CLIArgs

app = typer.Typer()


@app.command()
def main(
    file_string: Annotated[
        Optional[str],
        typer.Option(
            "--input-filestring", help="File content as a base64 encoded string."
        ),
    ] = None,
    file_url: Annotated[
        Optional[str], typer.Option("--input-fileurl", help="URL to an audio file.")
    ] = None,
    file_path: Annotated[
        Optional[Path], typer.Option("--input-filepath", help="Path to an audio file.")
    ] = None,
    num_speakers: Annotated[
        Optional[int], typer.Option("--num_speakers", help="Number of speakers.")
    ] = None,
    translate: Annotated[
        bool, typer.Option(help="Translate the audio to English.")
    ] = False,
    language: Annotated[
        Optional[str], typer.Option(help="Language of the audio.")
    ] = None,
    prompt: Annotated[
        Optional[str], typer.Option(help="Initial prompt for the model.")
    ] = None,
    preprocess: Annotated[
        int,
        typer.Option(help="Preprocessing level (0-4).", min=0, max=4, clamp=True),
    ] = 0,
    highpass_freq: Annotated[
        int, typer.Option("--highpass-freq", help="Highpass filter frequency.")
    ] = 45,
    lowpass_freq: Annotated[
        int, typer.Option("--lowpass-freq", help="Lowpass filter frequency.")
    ] = 8000,
    prop_decrease: Annotated[
        float, typer.Option("--prop-decrease", help="Proportion to decrease noise.")
    ] = 0.3,
    stationary: Annotated[
        bool, typer.Option(help="Whether the noise is stationary.")
    ] = True,
    target_dBFS: Annotated[
        float, typer.Option("--target-dBFS", help="Target dBFS for normalization.")
    ] = -18.0,
    device: Annotated[
        str, typer.Option(help="Device to use for computation (cpu or cuda).")
    ] = "cpu",
    compute_type: Annotated[
        str, typer.Option("--compute-type", help="Compute type for the model.")
    ] = "int8",
    model_name: Annotated[
        str, typer.Option("--model-name", help="Name of the Whisper model to use.")
    ] = "whisper-large-v3-turbo-et-verbatim-ct2",
    output_filename: Annotated[
        Optional[str],
        typer.Option("--output-filepath", help="Base name for the output JSON file."),
    ] = None,
) -> None:
    """
    Local runner for Whisper + Diarization.
    """
    logger.debug("Starting pipeline with provided arguments.")

    # Create CLI arguments model for type-safe configuration
    cli_args = CLIArgs(
        file_string=file_string,
        file_url=file_url,
        file_path=file_path,
        num_speakers=num_speakers,
        translate=translate,
        language=language,
        prompt=prompt,
        preprocess=preprocess,
        highpass_freq=highpass_freq,
        lowpass_freq=lowpass_freq,
        prop_decrease=prop_decrease,
        stationary=stationary,
        target_dBFS=target_dBFS,
        device=device,
        compute_type=compute_type,
        model_name=model_name,
        output_filename=output_filename,
    )

    # Create configuration from CLI arguments model
    config = WhisperDiarizationConfig.from_cli_args(cli_args)

    # Initialize pipeline with model configuration
    pipeline = WhisperDiarizationPipeline(config.model)

    # Run prediction with full configuration
    result = pipeline.predict(config)

    logger.info(result.model_dump_json(indent=2))

    # Handle output
    if config.output.output_filename:
        output_filename_base = config.output.output_filename
    else:
        output_filename_base = Path(__file__).parent / config.model.model_name

    write_json_file(
        output_filename_base=str(output_filename_base),
        content=result.model_dump(),
    )


if __name__ == "__main__":
    app()
