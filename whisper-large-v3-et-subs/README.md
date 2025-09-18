# whisper-large-v3-et-subs

This repository provides a script to transcribe audio files using the fine-tuned whisper-3-large V3 Estonian model or default whisper model.
You can run the script directly or via Docker.

See Hugging Face model: https://huggingface.co/TalTechNLP/whisper-large-v3-et-subs

## Prerequisites

- ffmpeg
- rust


## Build the Docker image - not working yet

```bash
docker build --pull -t whisper-large-et-subs .
```

## Run locally
```bash
# Run fine-tuned whisper-3-large with estonian subtitles
uv run main.py \
 -f tmpyegdo2jo.wav \
 -m taltechnlp/whisper-large-v3-et-subs \
 -p "Konsiori, kakskümmend üheksa, sada viiskümmend, sada kuuskümmend"

# Run default whisper-3-large
uv run main.py \
 -f tmpyegdo2jo.wav \
 -m openai/whisper-large-v3 \
 -p "Konsiori, kakskümmend üheksa, sada viiskümmend, sada kuuskümmend"
```

## Run the Docker container  - not working yet

```bash
docker run --rm \
  -v "$(pwd)":/app \
  whisper-large-et-subs tmpyegdo2jo.wav
```
