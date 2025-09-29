import orjson
from loguru import logger
from datetime import datetime
from pathlib import Path


def write_json_file(model_id: str, content: dict) -> None:
    """Generate a filename and write dictionary content to a JSON file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model_id = model_id.replace("/", "_")
    output_filename = f"{safe_model_id}_{timestamp}.json"
    output_path = Path(output_filename)

    with open(output_path, "wb") as f:
        f.write(
            orjson.dumps(
                content, option=orjson.OPT_INDENT_2 | orjson.OPT_SERIALIZE_NUMPY
            )
        )
    logger.info(f"Transcription saved to {output_path}")
