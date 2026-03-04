from voice_transcribe.config import PROCESS_PROMPT_FILE


def load_process_prompt(transcript: str) -> str:
    """Load the processing prompt template and substitute the transcript."""
    return PROCESS_PROMPT_FILE.read_text().replace("{transcript}", transcript)
