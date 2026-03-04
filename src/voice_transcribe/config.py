import pathlib

SAMPLE_RATE = 16000   # Hz — Whisper is trained on 16 kHz audio
CHANNELS = 1
DTYPE = "float32"
CHUNK_FRAMES = 512    # frames per read call (~32 ms at 16 kHz)
STREAM_DECODE_SECONDS = 1.0
STREAM_WINDOW_SECONDS = 5.0
STREAM_BUFFER_SECONDS = 12.0
MIN_STREAM_AUDIO_SECONDS = 0.8

# Model sizes: tiny | base | small | medium | large
DEFAULT_MODEL = "medium.en"
DEFAULT_MLX_MODEL = "mlx-community/whisper-small.en-mlx"

TRANSCRIPT_DIR = pathlib.Path.home() / "transcript"
IDLE_TIMEOUT = 300  # seconds of inactivity before auto-save and exit (5 minutes)

PROCESS_PROMPT_FILE = pathlib.Path(__file__).parent / "process_prompt.md"
DEFAULT_LLM_BACKEND = "claude"
DEFAULT_OLLAMA_MODEL = "qwen2.5:7b-instruct"

# Vector store (Phase 2)
VECTOR_DB_DIR = pathlib.Path.home() / ".voice_transcribe" / "index.lancedb"
VECTOR_SIMILARITY_THRESHOLD = 0.82
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # output dimension for all-MiniLM-L6-v2
