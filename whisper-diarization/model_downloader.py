#!/usr/bin/env uv run

import os

import typer
from loguru import logger
from typing_extensions import Annotated

from huggingface_hub import snapshot_download
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
from huggingface_hub.utils import GatedRepoError, HfHubHTTPError

from utils.token_helper import check_hf_token


def download_model(repo_id: str, local_name: str, subfolder: str = "ct2") -> str:
    """
    Downloads a model subfolder from the Hugging Face Hub and saves it with a specific local name.

    Args:
        repo_id (str): The repository ID of the model on the Hugging Face Hub.
        local_name (str): The desired local name for the model.
        subfolder (str): The subfolder to download from the repository (e.g., "ct2").
        hf_token (str, optional): Hugging Face token for private repositories. Defaults to None.

    Returns:
        str: The path to the downloaded model snapshot.
    """
    token = check_hf_token()

    cache_dir = HUGGINGFACE_HUB_CACHE
    model_dir = os.path.join(cache_dir, "models--" + repo_id.replace("/", "--"))

    # Use a safe version of the local name for the symlink
    safe_local_name = local_name.replace("/", "--")
    symlink_path = os.path.join(cache_dir, safe_local_name)

    try:
        logger.info(f"Downloading model {repo_id} to {model_dir}")
        model_snapshot_path = snapshot_download(
            repo_id=repo_id,
            allow_patterns=f"{subfolder}/*",
            cache_dir=cache_dir,
            token=token,
        )
        logger.info(f"Model {repo_id} downloaded to {model_snapshot_path}")

        # Create a symlink to the snapshot directory
        if not os.path.exists(symlink_path):
            os.symlink(model_snapshot_path, symlink_path)
            logger.info(f"Created symlink: {symlink_path} -> {model_snapshot_path}")
        else:
            logger.info(f"Symlink already exists: {symlink_path}")

        return model_snapshot_path

    except GatedRepoError:
        logger.error(
            f"Access to repository {repo_id} is denied. Please provide a valid Hugging Face token."
        )
        raise
    except HfHubHTTPError as e:
        logger.error(f"HTTP error while downloading model {repo_id}: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise


app = typer.Typer()


@app.command()
def main(
    repo_id: Annotated[
        str, typer.Option(help="The repository ID on the Hugging Face Hub.")
    ],
    local_name: Annotated[
        str, typer.Option("--local-name", help="The local name to use for the symlink.")
    ],
    subfolder: Annotated[
        str, typer.Option(help="The model subfolder to download (e.g., 'ct2').")
    ] = "ct2",
) -> None:
    """
    Downloads a model from the Hugging Face Hub and creates a local symlink.
    """
    # Validate required arguments
    if not repo_id:
        raise typer.BadParameter("repo_id is required")
    if not local_name:
        raise typer.BadParameter("local_name is required")

    logger.info(
        f"Received arguments: repo_id='{repo_id}', local_name='{local_name}', subfolder='{subfolder}'"
    )
    model_path = download_model(repo_id, local_name, subfolder)
    logger.info(f"Model '{repo_id}' downloaded successfully.")
    logger.info(f"   - Path: {model_path}")
    symlink_path = os.path.join(HUGGINGFACE_HUB_CACHE, local_name.replace("/", "--"))
    logger.info(f"   - Symlink: {symlink_path}")


if __name__ == "__main__":
    app()
