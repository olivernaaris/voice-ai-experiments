# How to run the project

This script downloads a model subfolder from the Hugging Face Hub and creates a local symlink in the Hugging Face cache directory. It is designed to fetch models, particularly those that require a specific subfolder (e.g., `ct2` models for CTranslate2).
And then execute the model from audio file to create diarized transcript.

## How to Run

Execute the script from your terminal, providing the repository ID and a local name for the model.

### Download ct2 compatible Esonian langauge fine-tuned fast-whisper model using custom CLI tool

```bash
./model_downloader.py \
--repo-id TalTechNLP/whisper-large-v3-turbo-et-verbatim \
--local-name whisper-large-v3-turbo-et-verbatim-ct2 \
--subfolder ct2
```

### Execute local model from cache directory to create transcription
```bash
./main.py \
--file_path tmpyegdo2jo.wav \
--device cpu \
--language et \
--model_name /Users/olivernaaris/.cache/huggingface/hub/whisper-large-v3-turbo-et-verbatim-ct2/ct2 \
--output_filename whisper-large-v3-turbo-et-verbatim-ct2
```
