import pathlib
import shutil
import subprocess


def render(syntax: str, output_path: pathlib.Path) -> dict:
    """Render D2 syntax to a PNG/SVG file via stdin.

    Uses `d2 - <output_path>` with syntax on stdin.
    Returns {"success": True, "image_path": str} or {"success": False, "error": str}.
    """
    d2_bin = shutil.which("d2")
    if d2_bin is None:
        return {"success": False, "error": "d2 is not installed (run: brew install d2)"}

    cmd = [d2_bin, "-", str(output_path)]
    print(f"  [d2] {' '.join(cmd)}", flush=True)
    print(f"  [d2 stdin]\n{syntax}", flush=True)

    try:
        result = subprocess.run(
            cmd,
            input=syntax,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return {"success": True, "image_path": str(output_path)}
        error = (result.stderr or result.stdout).strip()
        return {"success": False, "error": error or f"d2 exited with code {result.returncode}"}
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "d2 render timed out after 30s"}
    except Exception as exc:
        return {"success": False, "error": str(exc)}
