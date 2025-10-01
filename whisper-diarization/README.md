# Whisper Diarization Advanced

**Ultra-fast, customizable speech-to-text and speaker diarization for noisy, multi-speaker audio. Includes advanced noise reduction, stereo channel support, and flexible audio preprocessing—ideal for call centers, meetings, and podcasts.**

---

## Features

- **Ultra Fast & Cost-Effective:** Optimized for GPU/CPU environments.
- **Highly Customizable:** Choose your model, device, and audio preprocessing level.
- **Advanced Audio Treatment:** Built-in options for sanitization, high/low-pass filtering, aggressive noise reduction, and RMS normalization.
- **Stereo Channel Support:** Transcribes each channel separately for maximum speaker accuracy.
- **Multi-Input Flexibility:** Accepts direct file upload, URL, or base64 string.
- **Accurate Speaker Diarization & Transcription:** Utilizes state-of-the-art Whisper and Pyannote models.
- **Translation & Language Detection:** Automatically detects language and can translate speech to English.

---

## Getting Started

### 1. Download a Model

First, download a CTranslate2-compatible model using the provided script. For example, to download an Estonian model:

```bash
./model_downloader.py \
--repo-id TalTechNLP/whisper-large-v3-turbo-et-verbatim \
--local-name whisper-large-v3-turbo-et-verbatim-ct2 \
--subfolder ct2
```

### 2. Run Transcription

Execute the main script with your audio file and the path to the downloaded model.

```bash
./main.py \
--file_path tmpyegdo2jo.wav \
--device cpu \
--language et \
--preprocess 4 \
--num_speakers 3 \
--model_name /Users/olivernaaris/.cache/huggingface/hub/whisper-large-v3-turbo-et-verbatim-ct2/ct2 \
--output_filename whisper-large-v3-turbo-et-verbatim-ct2
```

---

## Documentation
- **[Architecture Overview](./docs/whisper-diarization-architecture.md):** A deep dive into the project's features, architecture, and configuration options.

## Thanks to
https://github.com/rafaelgalle/whisper-diarization-advanced/tree/main
