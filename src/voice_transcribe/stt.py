import pathlib
import tempfile
import wave
from collections.abc import Callable

import numpy as np
import torch
from faster_whisper import WhisperModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from voice_transcribe.config import CHANNELS, DEFAULT_MLX_MODEL, DEFAULT_MODEL, SAMPLE_RATE


def write_wav(path: pathlib.Path, audio: np.ndarray) -> None:
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())


def load_summariser():
    model_id = "philschmid/bart-large-cnn-samsum"
    print(
        f"Loading summarisation model '{model_id}' "
        "(first run will download the weights) ...",
        flush=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    if torch.cuda.is_available():
        model = model.to("cuda")
    print("Summarisation model ready.\n", flush=True)
    return model, tokenizer


def load_transcriber(
    stt_backend: str, faster_model_size: str = DEFAULT_MODEL, mlx_model_id: str = DEFAULT_MLX_MODEL
) -> Callable[[np.ndarray], list[str]]:
    if stt_backend == "mlx-whisper":
        try:
            import mlx_whisper
        except ImportError as exc:
            raise RuntimeError(
                "mlx-whisper backend selected but package is not installed. "
                "Install it with: uv pip install mlx-whisper"
            ) from exc

        mps_available = torch.backends.mps.is_available()
        print(
            f"Loading mlx-whisper model '{mlx_model_id}' on {'mps' if mps_available else 'cpu'} "
            "(first run may download weights) ...",
            flush=True,
        )

        def transcribe_chunk(audio_chunk: np.ndarray) -> list[str]:
            tmp_wav = pathlib.Path(tempfile.mktemp(suffix=".wav"))
            write_wav(tmp_wav, audio_chunk)
            try:
                result = mlx_whisper.transcribe(str(tmp_wav), path_or_hf_repo=mlx_model_id)
            finally:
                tmp_wav.unlink(missing_ok=True)
            text = result.get("text", "").strip() if isinstance(result, dict) else str(result).strip()
            return [text] if text else []

        print("mlx-whisper transcriber ready.\n", flush=True)
        return transcribe_chunk

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    print(
        f"Loading faster-whisper '{faster_model_size}' model on {device} "
        "(first run will download the weights) ...",
        flush=True,
    )
    model = WhisperModel(faster_model_size, device=device, compute_type=compute_type)
    print("faster-whisper model ready.\n", flush=True)

    def transcribe_chunk(audio_chunk: np.ndarray) -> list[str]:
        segs, _ = model.transcribe(
            audio_chunk,
            vad_filter=True,
            vad_parameters={"threshold": 0.5},
            no_speech_threshold=0.6,
            temperature=0.0,
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            condition_on_previous_text=False,
        )
        texts = []
        for seg in segs:
            text = seg.text.strip()
            if text:
                texts.append(text)
        return texts

    return transcribe_chunk
