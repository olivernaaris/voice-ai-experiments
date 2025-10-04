import os
import time
import torch
import shutil
import re
import pandas as pd
import numpy as np
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline as PyannotePipeline
from faster_whisper.vad import VadOptions

# Custom imports
from preprocess import preprocess_audio
from utils import (
    get_file,
    get_audio_channels,
    split_stereo_channels,
    logger,
    check_hf_token,
    diarize_audio,
)

# Custom schemas
from schema import DiarizationEntry, Output, Segment
from schema.config import ModelConfig, WhisperDiarizationConfig


class WhisperDiarizationPipeline:
    def __init__(
        self,
        model_config: ModelConfig | None = None,
    ) -> None:
        """Load models into memory."""

        if model_config is None:
            model_config = ModelConfig()

        logger.info(
            f"Setup with {model_config.model_name}, {model_config.device}, {model_config.compute_type}"
        )
        self.model = WhisperModel(
            model_size_or_path=model_config.model_name,
            device=model_config.device,
            compute_type=model_config.compute_type,
        )

        token = check_hf_token()
        logger.info("Using Hugging Face token for diarization model.")
        self.diarization_model = PyannotePipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1",
            token=token,
        ).to(torch.device(model_config.device))

    def predict(
        self,
        config: WhisperDiarizationConfig,
    ) -> Output:
        """Run a single prediction on the model."""
        # Get input file from configuration
        input_source = config.file_input.get_input_source()
        if not input_source:
            raise ValueError("No input source provided in configuration")

        temp_input, temp_dir = get_file(
            config.file_input.file_path,
            config.file_input.file_url,
            config.file_input.file_string,
        )

        try:
            num_channels = get_audio_channels(temp_input)
            logger.info(f"Audio with {num_channels} channels")
            if num_channels == 1:
                temp_processed = os.path.join(temp_dir, "input_processed.wav")
                if config.preprocessing.preprocess > 0:
                    preprocess_audio(
                        temp_input,
                        temp_processed,
                        preprocess_level=config.preprocessing.preprocess,
                        highpass_freq=config.preprocessing.highpass_freq,
                        lowpass_freq=config.preprocessing.lowpass_freq,
                        prop_decrease=config.preprocessing.prop_decrease,
                        stationary=config.preprocessing.stationary,
                        target_dBFS=config.preprocessing.target_dBFS,
                    )
                    audio_for_model = temp_processed
                else:
                    audio_for_model = temp_input

                logger.info("Starting transcribing mono")
                segments, detected_num_speakers, detected_language = (
                    self.speech_to_text(
                        audio_for_model,
                        config.transcription.num_speakers,
                        config.transcription.prompt or "",
                        config.transcription.language,
                        config.transcription.translate,
                    )
                )
                return Output(
                    segments=segments,
                    language=detected_language,
                    num_speakers=detected_num_speakers,
                )

            else:
                logger.info("Spliting channels")
                ch1_path, ch2_path = split_stereo_channels(temp_input, temp_dir)
                ch1_proc = os.path.join(temp_dir, "ch1_proc.wav")
                ch2_proc = os.path.join(temp_dir, "ch2_proc.wav")

                if config.preprocessing.preprocess > 0:
                    preprocess_audio(
                        ch1_path,
                        ch1_proc,
                        preprocess_level=config.preprocessing.preprocess,
                        highpass_freq=config.preprocessing.highpass_freq,
                        lowpass_freq=config.preprocessing.lowpass_freq,
                        prop_decrease=config.preprocessing.prop_decrease,
                        stationary=config.preprocessing.stationary,
                        target_dBFS=config.preprocessing.target_dBFS,
                    )
                    preprocess_audio(
                        ch2_path,
                        ch2_proc,
                        preprocess_level=config.preprocessing.preprocess,
                        highpass_freq=config.preprocessing.highpass_freq,
                        lowpass_freq=config.preprocessing.lowpass_freq,
                        prop_decrease=config.preprocessing.prop_decrease,
                        stationary=config.preprocessing.stationary,
                        target_dBFS=config.preprocessing.target_dBFS,
                    )
                else:
                    ch1_proc = ch1_path
                    ch2_proc = ch2_path

                logger.info("Starting transcribing stereo channel 0")
                # ch1_segments, info1 = self._transcribe_audio_ch0_mock(ch1_proc, language, prompt or "", translate)
                ch1_segments, info1 = self._transcribe_audio(
                    ch1_proc,
                    config.transcription.language,
                    config.transcription.prompt or "",
                    config.transcription.translate,
                )
                for s in ch1_segments:
                    s["speaker"] = "SPEAKER_00"
                    for w in s["words"]:
                        w["speaker"] = "SPEAKER_00"

                logger.info("Starting transcribing stereo channel 1")
                # ch2_segments, info2 = self._transcribe_audio_ch1_mock(ch2_proc, language, prompt or "", translate)
                ch2_segments, info2 = self._transcribe_audio(
                    ch2_proc,
                    config.transcription.language,
                    config.transcription.prompt or "",
                    config.transcription.translate,
                )
                for s in ch2_segments:
                    s["speaker"] = "SPEAKER_01"
                    for w in s["words"]:
                        w["speaker"] = "SPEAKER_01"
                # print(f"DEBUG --> Transcription stereo channel 1 {ch2_segments}")

                logger.info("Merging segments")
                # all_segments = sorted(ch1_segments + ch2_segments, key=lambda x: x["start"])
                all_segments = self.merge_stereo_words(ch1_segments, ch2_segments)

                detected_language = info1.language or info2.language
                return Output(
                    segments=all_segments, language=detected_language, num_speakers=2
                )

        except Exception as e:
            raise RuntimeError(f"Error running inference: {e}") from e

        finally:
            try:
                cleanup_candidates = {
                    locals().get("temp_input"),
                    locals().get("temp_processed"),
                    locals().get("ch1_path"),
                    locals().get("ch2_path"),
                    locals().get("ch1_proc"),
                    locals().get("ch2_proc"),
                }
                for f in cleanup_candidates:
                    if f and os.path.exists(f):
                        try:
                            os.remove(f)
                        except Exception:
                            pass
            except Exception:
                pass
            try:
                if temp_dir and os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass

    def speech_to_text(
        self,
        audio_file_wav: str,
        num_speakers: int | None = None,
        prompt: str = "",
        language: str | None = None,
        translate: bool = False,
    ) -> tuple[list[dict], int, str]:
        time_start = time.time()

        # segments, transcript_info = self._transcribe_audio_mock(
        segments, transcript_info = self._transcribe_audio(
            audio_file_wav, language, prompt, translate
        )
        logger.info(f"Finished transcribing, {len(segments)} segments")

        logger.info("Starting diarization")
        diarization, detected_num_speakers = diarize_audio(
            self.diarization_model, audio_file_wav, num_speakers
        )
        logger.info(f"Finished diarization, {detected_num_speakers} speakers detected")

        logger.info("Starting merging segments with speaker info")
        final_segments = self._merge_segments_with_diarization(segments, diarization)
        logger.info("Segments merged and cleaned")

        return final_segments, detected_num_speakers, transcript_info.language

    def _transcribe_audio(self, audio_file_wav, language, prompt, translate):
        options = dict(
            language=language,
            beam_size=5,
            vad_filter=True,
            vad_parameters=VadOptions(
                max_speech_duration_s=self.model.feature_extractor.chunk_length,
                min_speech_duration_ms=100,
                speech_pad_ms=100,
                threshold=0.25,
                neg_threshold=0.2,
            ),
            word_timestamps=True,
            initial_prompt=prompt,
            language_detection_segments=1,
            task="translate" if translate else "transcribe",
        )
        segments, transcript_info = self.model.transcribe(audio_file_wav, **options)
        segments = list(segments)
        segments = [
            {
                "avg_logprob": s.avg_logprob,
                "start": float(s.start),
                "end": float(s.end),
                "text": s.text,
                "words": [
                    {
                        "start": float(w.start),
                        "end": float(w.end),
                        "word": w.word,
                        "probability": w.probability,
                    }
                    for w in s.words
                ],
            }
            for s in segments
        ]
        return segments, transcript_info

    def _assign_speaker_to_segment_or_word(
        self, segment_or_word, diarize_df, fallback_speaker=None
    ):
        """Calculates the intersection of times."""
        diarize_df["intersection"] = np.minimum(
            diarize_df["end"], segment_or_word["end"]
        ) - np.maximum(diarize_df["start"], segment_or_word["start"])
        dia_tmp = diarize_df[diarize_df["intersection"] > 0]

        if len(dia_tmp) > 0:
            speaker = (
                dia_tmp.groupby("speaker")["intersection"]
                .sum()
                .sort_values(ascending=False)
                .index[0]
            )
        else:
            speaker = fallback_speaker or "UNKNOWN"
        return speaker

    def _merge_segments_with_diarization(
        self, segments: list[Segment], diarize_segments: list[DiarizationEntry]
    ) -> list[Segment]:
        diarize_df = pd.DataFrame(diarize_segments)
        final_segments = []

        for segment in segments:
            # Segment speaker
            speaker = self._assign_speaker_to_segment_or_word(segment, diarize_df)

            # Word-level speakers
            words_with_speakers = []
            for word in segment["words"]:
                word_speaker = self._assign_speaker_to_segment_or_word(
                    word, diarize_df, fallback_speaker=speaker
                )
                word["speaker"] = word_speaker
                words_with_speakers.append(word)

            new_segment = {
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"],
                "speaker": speaker,
                "avg_logprob": segment["avg_logprob"],
                "words": words_with_speakers,
            }
            final_segments.append(new_segment)

        final_segments = self._group_segments(final_segments)
        for segment in final_segments:
            segment["text"] = re.sub(r"\s+", " ", segment["text"]).strip()
            segment["text"] = re.sub(r"\s+([.,!?])", r"\1", segment["text"])
            segment["duration"] = segment["end"] - segment["start"]

        return final_segments

    def _group_segments(self, segments: list[Segment]) -> list[Segment]:
        if not segments:
            return []

        grouped_segments = []
        current_group = segments[0].copy()
        sentence_end_pattern = r"[.!?]+"

        for segment in segments[1:]:
            time_gap = segment["start"] - current_group["end"]
            current_duration = current_group["end"] - current_group["start"]
            can_combine = (
                segment["speaker"] == current_group["speaker"]
                and time_gap <= 1.0
                and current_duration < 30.0
                and not re.search(sentence_end_pattern, current_group["text"][-1:])
            )
            if can_combine:
                current_group["end"] = segment["end"]
                current_group["text"] += " " + segment["text"]
            else:
                grouped_segments.append(current_group)
                current_group = segment.copy()

        grouped_segments.append(current_group)
        return grouped_segments

    def merge_stereo_words(
        self,
        ch1_segments: list[Segment],
        ch2_segments: list[Segment],
        overlap_tolerance: float = 0.0,
        merge_margin: float = 1.0,
    ) -> list[Segment]:
        words = []
        for seg in ch1_segments + ch2_segments:
            for w in seg["words"]:
                words.append(
                    {
                        "start": w["start"],
                        "end": w["end"],
                        "word": w["word"],
                        "speaker": seg["speaker"],
                        "prob": w.get("probability", 1.0),
                    }
                )
        words = sorted(words, key=lambda x: x["start"])

        cleaned_words = []
        for i, w in enumerate(words):
            if cleaned_words:
                prev = cleaned_words[-1]
                if w["speaker"] != prev["speaker"] and w["start"] < prev["end"]:
                    overlap = prev["end"] - w["start"]
                    if overlap <= overlap_tolerance:
                        w["start"] = prev["end"] + 0.01
            cleaned_words.append(w)

        merged_segments = []
        current = None
        for w in cleaned_words:
            if not current:
                current = {
                    "start": w["start"],
                    "end": w["end"],
                    "speaker": w["speaker"],
                    "text": w["word"],
                    "words": [w],
                }
                continue

            if (
                w["speaker"] == current["speaker"]
                and w["start"] <= current["end"] + merge_margin
            ):
                current["end"] = w["end"]
                current["text"] += " " + w["word"]
                current["words"].append(w)
            else:
                merged_segments.append(current)
                current = {
                    "start": w["start"],
                    "end": w["end"],
                    "speaker": w["speaker"],
                    "text": w["word"],
                    "words": [w],
                }

        if current:
            merged_segments.append(current)

        return merged_segments
