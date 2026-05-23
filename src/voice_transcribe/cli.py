import argparse
import os
import sys

from voice_transcribe.audio import clear_screen
from voice_transcribe.config import (
    DEFAULT_GRANITE_MODEL,
    DEFAULT_LLM_BACKEND,
    DEFAULT_MLX_MODEL,
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_MODEL,
)
from voice_transcribe.loops import conversational
from voice_transcribe.storage import process_and_save
from voice_transcribe.stt import load_transcriber


def parse_args() -> argparse.Namespace:
    env_backend = os.getenv("VOICE_LLM_BACKEND", DEFAULT_LLM_BACKEND).lower()
    if env_backend not in {"claude", "ollama"}:
        env_backend = DEFAULT_LLM_BACKEND
    env_correction_backend = os.getenv("VOICE_CORRECTION_BACKEND", env_backend).lower()
    if env_correction_backend not in {"claude", "ollama"}:
        env_correction_backend = env_backend
    env_process_backend = os.getenv("VOICE_PROCESS_BACKEND", env_backend).lower()
    if env_process_backend not in {"claude", "ollama"}:
        env_process_backend = env_backend

    env_ollama_model = os.getenv("VOICE_OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    env_correction_model = os.getenv("VOICE_CORRECTION_OLLAMA_MODEL", env_ollama_model)
    env_process_model = os.getenv("VOICE_PROCESS_OLLAMA_MODEL", env_ollama_model)

    _DEFAULT_PRESET = (
        "--stt-backend faster-whisper --whisper-model medium.en "
        "--correction-backend ollama --process-backend ollama "
        "--correction-ollama-model qwen2.5:7b-instruct "
        "--process-ollama-model qwen2.5:7b-instruct"
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--default",
        action="store_true",
        default=False,
        help=(
            "Apply the built-in default preset, ignoring environment variables: "
            + _DEFAULT_PRESET
        ),
    )
    parser.add_argument(
        "--stt-backend",
        choices=["faster-whisper", "mlx-whisper", "granite-speech"],
        default=os.getenv("VOICE_STT_BACKEND", "faster-whisper"),
        help="Speech-to-text backend.",
    )
    parser.add_argument(
        "--whisper-model",
        default=os.getenv("VOICE_WHISPER_MODEL", DEFAULT_MODEL),
        help="faster-whisper model size to use when --stt-backend=faster-whisper.",
    )
    parser.add_argument(
        "--mlx-model",
        default=os.getenv("VOICE_MLX_MODEL", DEFAULT_MLX_MODEL),
        help="mlx-whisper model id to use when --stt-backend=mlx-whisper.",
    )
    parser.add_argument(
        "--granite-model",
        default=os.getenv("VOICE_GRANITE_MODEL", DEFAULT_GRANITE_MODEL),
        help="Granite Speech model id to use when --stt-backend=granite-speech.",
    )
    parser.add_argument(
        "--correction-backend",
        choices=["claude", "ollama"],
        default=env_correction_backend,
        help="LLM backend for correction.",
    )
    parser.add_argument(
        "--process-backend",
        choices=["claude", "ollama"],
        default=env_process_backend,
        help="LLM backend for final processing/saving.",
    )
    parser.add_argument(
        "--correction-ollama-model",
        default=env_correction_model,
        help="Ollama model to use when --correction-backend=ollama.",
    )
    parser.add_argument(
        "--process-ollama-model",
        default=env_process_model,
        help="Ollama model to use when --process-backend=ollama.",
    )
    args = parser.parse_args()

    if args.default:
        args.stt_backend = "faster-whisper"
        args.whisper_model = DEFAULT_MODEL
        args.correction_backend = "ollama"
        args.process_backend = "ollama"
        args.correction_ollama_model = DEFAULT_OLLAMA_MODEL
        args.process_ollama_model = DEFAULT_OLLAMA_MODEL

    return args


def _print_config(args: argparse.Namespace) -> None:
    print("Effective configuration:")
    print(f"  --stt-backend               {args.stt_backend}")
    print(f"  --whisper-model             {args.whisper_model}")
    print(f"  --mlx-model                 {args.mlx_model}")
    print(f"  --granite-model             {args.granite_model}")
    print(f"  --correction-backend        {args.correction_backend}")
    print(f"  --correction-ollama-model   {args.correction_ollama_model}")
    print(f"  --process-backend           {args.process_backend}")
    print(f"  --process-ollama-model      {args.process_ollama_model}")
    print()
    print("Use --default to apply the built-in preset (ignores env vars):")
    print("  --stt-backend faster-whisper --whisper-model medium.en")
    print("  --correction-backend ollama  --correction-ollama-model qwen2.5:7b-instruct")
    print("  --process-backend ollama     --process-ollama-model qwen2.5:7b-instruct")
    print()
    print("Run 'voice --help' for all options.")


def main() -> None:
    # sys.stderr = open(os.devnull, "w")  # uncomment once stable
    args = parse_args()

    if len(sys.argv) == 1:
        _print_config(args)
        sys.exit(0)

    transcribe_chunk_fn = load_transcriber(
        stt_backend=args.stt_backend,
        faster_model_size=args.whisper_model,
        mlx_model_id=args.mlx_model,
        granite_model_id=args.granite_model,
    )
    clear_screen()
    print(f"STT backend: {args.stt_backend}", flush=True)
    if args.stt_backend == "faster-whisper":
        print(f"Whisper model: {args.whisper_model}", flush=True)
    elif args.stt_backend == "mlx-whisper":
        print(f"MLX model: {args.mlx_model}", flush=True)
    elif args.stt_backend == "granite-speech":
        print(f"Granite model: {args.granite_model}", flush=True)
    print(f"Correction backend: {args.correction_backend}", flush=True)
    if args.correction_backend == "ollama":
        print(f"Correction model: {args.correction_ollama_model}", flush=True)
    print(f"Process backend: {args.process_backend}", flush=True)
    if args.process_backend == "ollama":
        print(f"Process model: {args.process_ollama_model}", flush=True)

    text = conversational.run(
        transcribe_chunk_fn,
        args.correction_backend,
        args.correction_ollama_model,
    )
    if text:
        process_and_save(text, args.process_backend, args.process_ollama_model)
    sys.exit(0)
