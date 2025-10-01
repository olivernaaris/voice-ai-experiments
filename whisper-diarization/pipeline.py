from __future__ import annotations
import os
import time
import torch
import torchaudio
import shutil
import re
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline as PyannotePipeline
from faster_whisper.vad import VadOptions

# Custom imports
from preprocess import preprocess_audio
from audio_utils import get_file, get_audio_channels, split_stereo_channels

# Custom schemas
from schema.outputs import Output


class WhisperDiarizationPipeline:
    def __init__(
        self,
        device: str = "cpu",
        compute_type: str = "int8",
        model_name: str = "large-v3-turbo",
    ) -> None:
        """Load models into memory."""

        print(f"DEBUG --> Setup with {model_name}, {device}, {compute_type}")
        self.model = WhisperModel(
            model_size_or_path=model_name,
            device=device,
            compute_type=compute_type,
        )
        token = os.getenv("HF_TOKEN") or os.getenv("HF_AUTH_TOKEN")
        if token is None:
            raise ValueError(
                "Hugging Face token not found. Please set the HF_TOKEN environment variable."
            )
        print("DEBUG --> Using Hugging Face token for diarization model.")
        self.diarization_model = PyannotePipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=token,
        ).to(torch.device(device))

    def predict(
        self,
        file_string: Optional[str] = None,
        file_url: Optional[str] = None,
        file_path: Optional[str] = None,
        num_speakers: Optional[int] = None,
        translate: bool = False,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        preprocess: int = 4,
        highpass_freq: int = 45,
        lowpass_freq: int = 8000,
        prop_decrease: float = 1.0,
        stationary: bool = True,
        target_dBFS: float = -18.0,
    ) -> Output:
        """Run a single prediction on the model."""
        temp_input, temp_dir = get_file(file_path, file_url, file_string)

        try:
            num_channels = get_audio_channels(temp_input)
            print(f"DEBUG --> Audio with {num_channels} channels")
            if num_channels == 1:
                temp_processed = os.path.join(temp_dir, "input_processed.wav")
                if preprocess > 0:
                    preprocess_audio(
                        temp_input,
                        temp_processed,
                        preprocess_level=preprocess,
                        highpass_freq=highpass_freq,
                        lowpass_freq=lowpass_freq,
                        prop_decrease=prop_decrease,
                        stationary=stationary,
                        target_dBFS=target_dBFS,
                    )
                    audio_for_model = temp_processed
                else:
                    audio_for_model = temp_input

                print("DEBUG --> Starting transcribing mono")
                segments, detected_num_speakers, detected_language = (
                    self.speech_to_text(
                        audio_for_model, num_speakers, prompt or "", language, translate
                    )
                )
                return Output(segments, detected_language, detected_num_speakers)

            else:
                print("DEBUG --> Spliting channels")
                ch1_path, ch2_path = split_stereo_channels(temp_input, temp_dir)
                ch1_proc = os.path.join(temp_dir, "ch1_proc.wav")
                ch2_proc = os.path.join(temp_dir, "ch2_proc.wav")

                if preprocess > 0:
                    preprocess_audio(
                        ch1_path,
                        ch1_proc,
                        preprocess_level=preprocess,
                        highpass_freq=highpass_freq,
                        lowpass_freq=lowpass_freq,
                        prop_decrease=prop_decrease,
                        stationary=stationary,
                        target_dBFS=target_dBFS,
                    )
                    preprocess_audio(
                        ch2_path,
                        ch2_proc,
                        preprocess_level=preprocess,
                        highpass_freq=highpass_freq,
                        lowpass_freq=lowpass_freq,
                        prop_decrease=prop_decrease,
                        stationary=stationary,
                        target_dBFS=target_dBFS,
                    )
                else:
                    ch1_proc = ch1_path
                    ch2_proc = ch2_path

                print("DEBUG --> Starting transcribing stereo channel 0")
                # ch1_segments, info1 = self._transcribe_audio_ch0_mock(ch1_proc, language, prompt or "", translate)
                ch1_segments, info1 = self._transcribe_audio(
                    ch1_proc, language, prompt or "", translate
                )
                for s in ch1_segments:
                    s["speaker"] = "SPEAKER_00"
                    for w in s["words"]:
                        w["speaker"] = "SPEAKER_00"
                # print(f"DEBUG --> Transcription stereo channel 0 {ch1_segments}")

                print("DEBUG --> Starting transcribing stereo channel 1")
                # ch2_segments, info2 = self._transcribe_audio_ch1_mock(ch2_proc, language, prompt or "", translate)
                ch2_segments, info2 = self._transcribe_audio(
                    ch2_proc, language, prompt or "", translate
                )
                for s in ch2_segments:
                    s["speaker"] = "SPEAKER_01"
                    for w in s["words"]:
                        w["speaker"] = "SPEAKER_01"
                # print(f"DEBUG --> Transcription stereo channel 1 {ch2_segments}")

                print("DEBUG --> Merging segments")
                # all_segments = sorted(ch1_segments + ch2_segments, key=lambda x: x["start"])
                all_segments = self.merge_stereo_words(ch1_segments, ch2_segments)

                detected_language = info1.language or info2.language
                return Output(all_segments, detected_language, 2)

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
                # if 'temp_dir' in locals() and temp_dir:
                #     temp_dir.cleanup()
            except Exception:
                pass

    def speech_to_text(
        self,
        audio_file_wav: str,
        num_speakers: Optional[int] = None,
        prompt: str = "",
        language: Optional[str] = None,
        translate: bool = False,
    ) -> Tuple[List[Dict], int, str]:
        time_start = time.time()

        # segments, transcript_info = self._transcribe_audio_mock(
        segments, transcript_info = self._transcribe_audio(
            audio_file_wav, language, prompt, translate
        )
        print(f"DEBUG --> Finished transcribing, {len(segments)} segments")

        print("DEBUG --> Starting diarization")
        # diarization, detected_num_speakers = self._diarize_audio_mock(
        diarization, detected_num_speakers = self._diarize_audio(
            audio_file_wav, num_speakers
        )
        print(
            f"DEBUG --> Finished diarization, {detected_num_speakers} speakers detected"
        )

        print("DEBUG --> Starting merging segments with speaker info")
        final_segments = self._merge_segments_with_diarization(segments, diarization)
        print("DEBUG --> Segments merged and cleaned")

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

    def _diarize_audio(self, audio_file_wav, num_speakers=None):
        waveform, sample_rate = torchaudio.load(audio_file_wav)
        diarization = self.diarization_model(
            {"waveform": waveform, "sample_rate": sample_rate},
            num_speakers=num_speakers,
        )

        diarize_segments = []
        diarization_list = list(diarization.itertracks(yield_label=True))
        for turn, _, speaker in diarization_list:
            diarize_segments.append(
                {"start": turn.start, "end": turn.end, "speaker": speaker}
            )

        unique_speakers = {speaker for _, _, speaker in diarization_list}
        detected_num_speakers = len(unique_speakers)

        return diarize_segments, detected_num_speakers

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

    def _merge_segments_with_diarization(self, segments, diarize_segments):
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

    def _group_segments(self, segments):
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
        self, ch1_segments, ch2_segments, overlap_tolerance=0, merge_margin=1
    ):
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
