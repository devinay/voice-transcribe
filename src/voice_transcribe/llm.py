import pathlib
import subprocess
import tempfile


def run_ollama_prompt(prompt: str, model: str) -> str:
    """Run an Ollama prompt and return stdout."""
    result = subprocess.run(
        ["ollama", "run", model, prompt],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def run_claude_prompt(prompt: str) -> str:
    """Run a Claude prompt and return stdout."""
    result = subprocess.run(
        ["claude", "-p", prompt],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def run_llm_prompt(prompt: str, llm_backend: str, ollama_model: str) -> str:
    """Run prompt with selected backend and return stdout."""
    if llm_backend == "ollama":
        return run_ollama_prompt(prompt, ollama_model)
    return run_claude_prompt(prompt)


def correct_with_llm(
    transcript: str, instructions: str, llm_backend: str, ollama_model: str
) -> str:
    """Apply spoken correction instructions to the transcript using the selected LLM backend."""
    if llm_backend == "ollama":
        prompt = (
            "You are editing a transcript.\n"
            "Apply the user's correction instructions to the transcript text.\n"
            "Return only the corrected transcript text, with no commentary.\n\n"
            f"Correction instructions:\n{instructions}\n\n"
            f"Transcript:\n{transcript}\n"
        )
        return run_ollama_prompt(prompt, ollama_model)

    # Default: claude CLI path
    tmp = pathlib.Path(tempfile.mktemp(suffix=".md"))
    tmp.write_text(transcript)
    subprocess.run(
        [
            "claude", "-p",
            f"Edit the file {tmp} — apply these correction instructions to the transcript "
            f"text. Only the corrected transcript text should remain in the file, no "
            f"commentary: {instructions}",
            "--allowedTools", "Edit,Write,Read",
        ],
        check=True,
        stderr=subprocess.DEVNULL,
    )
    corrected = tmp.read_text().strip()
    tmp.unlink()
    return corrected
