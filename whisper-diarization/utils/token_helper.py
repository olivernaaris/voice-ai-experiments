import os
from typing import Optional


def check_hf_token(token: Optional[str] = None) -> str:
    """
    Checks for the Hugging Face token in the environment variables.

    This function is used to ensure that the Hugging Face token is available
    for operations that require authentication, such as downloading models
    from private repositories.

    Args:
        token (Optional[str]): An optional token to check. If provided,
                               it will be returned directly.

    Returns:
        str: The Hugging Face token.

    Raises:
        ValueError: If the token is not provided and the HF_TOKEN
                    environment variable is not set.
    """
    if token:
        return token

    env_token = os.getenv("HF_TOKEN")
    if env_token:
        return env_token

    raise ValueError("HF_TOKEN environment variable not set.")
