import json
import os
import pathlib
import subprocess
import tempfile
import urllib.request


def run_ollama_prompt(prompt: str, model: str) -> str:
    """Run an Ollama prompt via HTTP API (keeps model warm between calls)."""
    payload = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode()
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    return data.get("response", "").strip()


def run_claude_prompt(prompt: str) -> str:
    """Run a Claude prompt via Anthropic SDK."""
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text.strip()


def run_openai_prompt(prompt: str) -> str:
    """Run a prompt via OpenAI API."""
    from openai import OpenAI

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2048,
    )
    return response.choices[0].message.content.strip()


def run_llm_prompt(prompt: str, llm_backend: str, ollama_model: str) -> str:
    """Run prompt with selected backend and return response text."""
    if llm_backend == "ollama":
        return run_ollama_prompt(prompt, ollama_model)
    if llm_backend == "openai":
        return run_openai_prompt(prompt)
    return run_claude_prompt(prompt)


def correct_with_llm(
    transcript: str, instructions: str, llm_backend: str, ollama_model: str
) -> str:
    """Apply spoken correction instructions to the transcript using the selected LLM backend."""
    prompt = (
        "You are editing a transcript.\n"
        "Apply the user's correction instructions to the transcript text.\n"
        "Return only the corrected transcript text, with no commentary.\n\n"
        f"Correction instructions:\n{instructions}\n\n"
        f"Transcript:\n{transcript}\n"
    )
    if llm_backend == "ollama":
        return run_ollama_prompt(prompt, ollama_model)
    if llm_backend == "openai":
        return run_openai_prompt(prompt)

    # claude: use file-based edit for large transcripts
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
