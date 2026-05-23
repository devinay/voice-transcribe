import pathlib
import tempfile
import wave
from collections.abc import Callable

import numpy as np
import torch
from faster_whisper import WhisperModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from voice_transcribe.config import CHANNELS, DEFAULT_GRANITE_MODEL, DEFAULT_MLX_MODEL, DEFAULT_MODEL, SAMPLE_RATE


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
    stt_backend: str,
    faster_model_size: str = DEFAULT_MODEL,
    mlx_model_id: str = DEFAULT_MLX_MODEL,
    granite_model_id: str = DEFAULT_GRANITE_MODEL,
) -> Callable[[np.ndarray], list[str]]:
    if stt_backend == "granite-speech":
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        if torch.backends.mps.is_available():
            device = "mps"
            torch_dtype = torch.float16
        elif torch.cuda.is_available():
            device = "cuda"
            torch_dtype = torch.bfloat16
        else:
            device = "cpu"
            torch_dtype = torch.float32

        print(
            f"Loading granite-speech model '{granite_model_id}' on {device} "
            "(first run will download weights ~4GB) ...",
            flush=True,
        )
        processor = AutoProcessor.from_pretrained(granite_model_id)
        tokenizer = processor.tokenizer
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            granite_model_id, dtype=torch_dtype
        ).to(device)

        chat = [{"role": "user", "content": "<|audio|>transcribe the speech with proper punctuation and capitalization."}]
        prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

        print("granite-speech transcriber ready.\n", flush=True)

        def transcribe_chunk(audio_chunk: np.ndarray) -> list[str]:
            wav = torch.from_numpy(audio_chunk.flatten().astype(np.float32)).unsqueeze(0)
            model_inputs = processor(prompt, wav, device=device, return_tensors="pt").to(device)
            with torch.no_grad():
                model_outputs = model.generate(**model_inputs, max_new_tokens=200, do_sample=False)
            num_input_tokens = model_inputs["input_ids"].shape[-1]
            new_tokens = model_outputs[0, num_input_tokens:].unsqueeze(0)
            text = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
            return [text] if text else []

        return transcribe_chunk

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
