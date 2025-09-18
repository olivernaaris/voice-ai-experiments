FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

# UV_COMPILE_BYTECODE=1: Compile Python files to .pyc for faster startup times
# UV_LINK_MODE=copy: Copy packages from wheel cache (safer for containers)
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
# UV_PYTHON_DOWNLOADS=0: Disable automatic Python downloads (use container's Python)
ENV UV_PYTHON_DOWNLOADS=0

# Create and set working directory
RUN mkdir -p /app
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    ffmpeg \
    build-essential \
    ca-certificates \
    rustc \
    cargo \
  && rm -rf /var/lib/apt/lists/*

# Copy dependency files first (for better Docker layer caching)
COPY pyproject.toml uv.lock ./

# Copy the main script
COPY main.py ./

ENTRYPOINT ["uv", "run", "main.py"]
CMD []
