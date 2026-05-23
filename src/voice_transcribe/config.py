import pathlib

SAMPLE_RATE = 16000   # Hz
CHANNELS = 1
DTYPE = "float32"
CHUNK_FRAMES = 512    # frames per read call (~32 ms at 16 kHz)

TRANSCRIPT_DIR = pathlib.Path.home() / "transcript"
IDLE_TIMEOUT = 300  # seconds of inactivity before auto-exit

PROCESS_PROMPT_FILE = pathlib.Path(__file__).parent / "process_prompt.md"
DEFAULT_LLM_BACKEND = "claude"
DEFAULT_OLLAMA_MODEL = "qwen2.5:7b-instruct"

# Deepgram
DEEPGRAM_API_KEY_ENV = "DEEPGRAM_API_KEY"
DEEPGRAM_MODEL_ENV = "VOICE_DEEPGRAM_MODEL"

# Protocol
PROTOCOL_REPAIR_RETRIES = 2

# Vector store (Phase 2)
VECTOR_DB_DIR = pathlib.Path.home() / ".voice_transcribe" / "index.lancedb"
LOG_FILE = pathlib.Path.home() / ".voice_transcribe" / "output.log"
VECTOR_SIMILARITY_THRESHOLD = 0.82
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # output dimension for all-MiniLM-L6-v2
