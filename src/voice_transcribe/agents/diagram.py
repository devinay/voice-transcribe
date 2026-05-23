import pathlib

from voice_transcribe.loops.agent import run as agent_run
from voice_transcribe.tools.d2 import render as d2_render

_SYSTEM_PROMPT = """You are a diagram generation assistant. Your job is to produce valid D2 diagram syntax.

D2 syntax examples:

# Simple flow
direction: right
client -> server: request
server -> db: query
db: {shape: cylinder}

# Grouped nodes
direction: right
backend: {
  api
  worker
}
client -> backend.api: call
backend.api -> backend.worker: dispatch

# Sequence diagram
conversation: {
  shape: sequence_diagram
  user -> llm: prompt
  llm -> user: response
}

Rules:
  - Write plain D2 syntax only — no section headers, no labels like "nodes:" or "edges:".
  - Keep diagrams focused: 5-12 nodes max.
  - Iterate on errors from the tool until the render succeeds.
  - When the render succeeds, stop calling tools and output only: DONE"""

_TOOLS = [
    {
        "name": "render_d2",
        "description": "Render D2 syntax to a PNG file. Returns success/error.",
        "parameters": {
            "syntax": "string — complete D2 diagram source"
        },
    }
]


def run(
    description: str,
    output_path: pathlib.Path,
    llm_backend: str,
    ollama_model: str,
    log=None,
) -> dict:
    """Run the diagram agent for a single diagram description.

    Returns the result dict from d2_render on success, or {"success": False, ...} on failure.
    log: optional callable(str) for per-session logging.
    """
    last_result: dict = {"success": False, "error": "agent produced no render call"}
    last_syntax: list[str] = [""]
    attempt: list[int] = [0]

    def _emit(msg: str) -> None:
        print(msg, flush=True)
        if log:
            log(msg)

    def _render_d2(syntax: str) -> dict:
        nonlocal last_result
        attempt[0] += 1
        last_syntax[0] = syntax
        _emit(f"  [attempt {attempt[0]}] rendering {len(syntax)}-char D2 syntax...")
        last_result = d2_render(syntax, output_path)
        if last_result.get("success"):
            _emit(f"  [attempt {attempt[0]}] render ok")
        else:
            _emit(f"  [attempt {attempt[0]}] error: {last_result.get('error', '')[:200]}")
        return last_result

    MAX_ATTEMPTS = 3

    agent_run(
        system_prompt=_SYSTEM_PROMPT,
        initial_message=f"Generate a D2 diagram: {description}",
        tools=_TOOLS,
        tool_fns={"render_d2": _render_d2},
        llm_backend=llm_backend,
        ollama_model=ollama_model,
        max_iterations=MAX_ATTEMPTS,
    )

    if not last_result.get("success") and attempt[0] >= MAX_ATTEMPTS:
        last_err = last_result.get("error", "unknown")
        msg = (
            f"  [diagram] gave up after {MAX_ATTEMPTS} attempts"
            f" — last error: {last_err}"
        )
        _emit(msg)

    return {**last_result, "syntax": last_syntax[0]}
