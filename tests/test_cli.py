import importlib


def _load_cli():
    return importlib.import_module("voice_transcribe.cli")


def test_parse_args_default_preset_overrides_env(monkeypatch):
    cli = _load_cli()

    monkeypatch.setenv("VOICE_STT_BACKEND", "mlx-whisper")
    monkeypatch.setenv("VOICE_WHISPER_MODEL", "small.en")
    monkeypatch.setenv("VOICE_LLM_BACKEND", "claude")
    monkeypatch.setenv("VOICE_CORRECTION_BACKEND", "claude")
    monkeypatch.setenv("VOICE_PROCESS_BACKEND", "claude")
    monkeypatch.setenv("VOICE_CORRECTION_OLLAMA_MODEL", "not-used")
    monkeypatch.setenv("VOICE_PROCESS_OLLAMA_MODEL", "not-used")
    monkeypatch.setattr("sys.argv", ["voice", "--default"])

    args = cli.parse_args()

    assert args.stt_backend == "faster-whisper"
    assert args.whisper_model == cli.DEFAULT_MODEL
    assert args.correction_backend == "ollama"
    assert args.process_backend == "ollama"
    assert args.correction_ollama_model == cli.DEFAULT_OLLAMA_MODEL
    assert args.process_ollama_model == cli.DEFAULT_OLLAMA_MODEL


def test_parse_args_uses_llm_backend_as_fallback(monkeypatch):
    cli = _load_cli()

    monkeypatch.setenv("VOICE_LLM_BACKEND", "ollama")
    monkeypatch.delenv("VOICE_CORRECTION_BACKEND", raising=False)
    monkeypatch.delenv("VOICE_PROCESS_BACKEND", raising=False)
    monkeypatch.setenv("VOICE_OLLAMA_MODEL", "qwen2.5:7b-instruct")
    monkeypatch.delenv("VOICE_CORRECTION_OLLAMA_MODEL", raising=False)
    monkeypatch.delenv("VOICE_PROCESS_OLLAMA_MODEL", raising=False)
    monkeypatch.setattr("sys.argv", ["voice", "--stt-backend", "faster-whisper"])

    args = cli.parse_args()

    assert args.correction_backend == "ollama"
    assert args.process_backend == "ollama"
    assert args.correction_ollama_model == "qwen2.5:7b-instruct"
    assert args.process_ollama_model == "qwen2.5:7b-instruct"
