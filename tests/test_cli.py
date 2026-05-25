import importlib
import sys


def _load_cli():
    return importlib.import_module("voice_transcribe.cli")


def test_parse_args_defaults(monkeypatch):
    cli = _load_cli()
    monkeypatch.delenv("VOICE_LLM_BACKEND", raising=False)
    monkeypatch.delenv("VOICE_OLLAMA_MODEL", raising=False)
    monkeypatch.setattr("sys.argv", ["voice", "--llm-backend", "claude"])

    args = cli.parse_args()

    assert args.llm_backend == "claude"
    assert args.ollama_model == "qwen2.5:7b-instruct"


def test_parse_args_env_overrides(monkeypatch):
    cli = _load_cli()
    monkeypatch.setenv("VOICE_LLM_BACKEND", "ollama")
    monkeypatch.setenv("VOICE_OLLAMA_MODEL", "llama3")
    monkeypatch.setattr("sys.argv", ["voice", "--llm-backend", "ollama"])

    args = cli.parse_args()

    assert args.llm_backend == "ollama"
    assert args.ollama_model == "llama3"


def test_get_api_key_missing(monkeypatch):
    cli = _load_cli()
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)

    with_exit = False
    try:
        cli._get_api_key()
    except SystemExit:
        with_exit = True

    assert with_exit, "should exit when DEEPGRAM_API_KEY is not set"


def test_get_api_key_present(monkeypatch):
    cli = _load_cli()
    monkeypatch.setenv("DEEPGRAM_API_KEY", "test-key-123")

    key = cli._get_api_key()
    assert key == "test-key-123"
