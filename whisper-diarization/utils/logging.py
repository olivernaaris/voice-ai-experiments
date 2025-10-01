import sys

from loguru import logger as loguru_logger

loguru_logger.remove()
loguru_logger.add(sys.stderr, level="INFO")

logger = loguru_logger

__all__ = ["logger"]
