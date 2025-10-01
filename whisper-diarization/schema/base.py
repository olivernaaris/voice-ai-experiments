"""Base schema definitions for the application."""

from pydantic import BaseModel


class CustomBaseModel(BaseModel):
    """A custom base model that uses orjson for serialization."""

    pass
