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

### 1. Manually approve the following Pyannote Hugging Face models so we can download them
- https://huggingface.co/pyannote/segmentation-3.0
- https://huggingface.co/pyannote/speaker-diarization-3.1
- https://huggingface.co/pyannote/speaker-diarization-community-1


### 2. Download custom fine-tuned Whisper-3 model from Hugging Face

First, download a CTranslate2-compatible model using the provided script. For example, to download an Estonian model:

```bash
./model_downloader.py \
--repo-id TalTechNLP/whisper-large-v3-turbo-et-verbatim \
--local-name whisper-large-v3-turbo-et-verbatim-ct2 \
--subfolder ct2
```

### 3. Run Transcription and Diarization

Execute the main script with your audio file and the path to the downloaded model.

```bash
./main.py \
--input-filepath audio_files/2min-wav-et.wav \
--device cpu \
--language et \
--preprocess 4 \
--model-name /Users/olivernaaris/.cache/huggingface/hub/whisper-large-v3-turbo-et-verbatim-ct2/ct2 \
--output-filepath evaluation/whisper-large-v3-turbo-et-verbatim-ct2
```

### 4. Run Transcription and Diarization profiler to time execution
```bash
uv run python -m cProfile -s time \
./main.py \
--input-filepath audio_files/2min-wav-et.wav \
--device cpu \
--language et \
--preprocess 4 \
--model-name /Users/olivernaaris/.cache/huggingface/hub/whisper-large-v3-turbo-et-verbatim-ct2/ct2 \
--output-filepath evaluation/whisper-large-v3-turbo-et-verbatim-ct2
```


## Documentation
- **[Architecture Overview](./docs/whisper-diarization-architecture.md):** A deep dive into the project's features, architecture, and configuration options.
- **[ASR Evaluation](./docs/asr-evaluation-guide.md):** How to evaluate the ASR models output error rates.
- **[Please read](https://medium.com/@rafaelgalle1/building-a-custom-scalable-audio-transcription-pipeline-whisper-pyannote-ffmpeg-d0f03f884330)

## Thanks to
https://github.com/rafaelgalle/whisper-diarization-advanced/tree/main
