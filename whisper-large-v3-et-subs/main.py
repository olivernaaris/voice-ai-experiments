import os

os.environ.setdefault(
    "HF_HUB_ENABLE_HF_TRANSFER", "1"
)  # enable hf-transfer by default for faster downloads

import typer
import orjson
import time
from datetime import datetime
from pathlib import Path
from loguru import logger
from typing_extensions import Annotated
from enum import Enum

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from huggingface_hub.errors import HfHubHTTPError


class CliModelId(str, Enum):
    WHISPER_LARGE_V3 = "openai/whisper-large-v3"
    WHISPER_LARGE_V3_ET_SUBS = "TalTechNLP/whisper-large-v3-et-subs"


app = typer.Typer()


def process_audio(file_path: Path, model_id: str) -> None:
    """Transcribe a single audio file and save the result to a JSON file."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    try:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, dtype=dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
    except HfHubHTTPError:
        logger.error("Connection error: The model download failed. ")
        raise
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        dtype=dtype,
        device=device,
        return_timestamps=True,
    )

    start_time = time.perf_counter()
    result = pipe(
        str(file_path), generate_kwargs={"task": "transcribe", "language": "et"}
    )
    end_time = time.perf_counter()
    inference_time = end_time - start_time
    logger.info(
        f"Inference took {inference_time:.4f} seconds for {model_id} on {device}"
    )

    write_json_file(model_id=model_id, content=result)


def write_json_file(model_id: str, content: dict) -> None:
    """Generate a filename and write dictionary content to a JSON file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model_id = model_id.replace("/", "_")
    output_filename = f"{safe_model_id}_{timestamp}.json"
    output_path = Path(output_filename)

    with open(output_path, "wb") as f:
        f.write(orjson.dumps(content, option=orjson.OPT_INDENT_2))
    logger.info(f"Transcription saved to {output_path}")


@app.command()
def parse(
    file_path: Annotated[
        Path,
        typer.Option(
            "--file-path",
            "-f",
            help="Path to the audio file to transcribe.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    model_id: Annotated[
        CliModelId,
        typer.Option(
            "--model-id",
            "-m",
            help="The STT model to use for transcription.",
            case_sensitive=False,
            show_default=True,
        ),
    ],
) -> None:
    """
    Transcribe an audio file.
    """
    process_audio(file_path, model_id.value)


if __name__ == "__main__":
    app()
