import difflib

_RED        = "\033[31m"
_STRIKE_RED = "\033[31m\033[9m"
_GREEN      = "\033[32m"
_RESET      = "\033[0m"
_CLEAR      = "\r" + " " * 40 + "\r"


def clear_screen() -> None:
    print("\033[2J\033[H", end="", flush=True)


def highlight_diff(before: str, after: str) -> str:
    """Return after text with removed words in red strikethrough and added words in green."""
    before_words = before.split()
    after_words = after.split()
    matcher = difflib.SequenceMatcher(None, before_words, after_words)
    result = []
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            result.append(" ".join(after_words[j1:j2]))
        elif op == "replace":
            result.append(f"{_STRIKE_RED}{' '.join(before_words[i1:i2])}{_RESET}")
            result.append(f"{_GREEN}{' '.join(after_words[j1:j2])}{_RESET}")
        elif op == "delete":
            result.append(f"{_STRIKE_RED}{' '.join(before_words[i1:i2])}{_RESET}")
        elif op == "insert":
            result.append(f"{_GREEN}{' '.join(after_words[j1:j2])}{_RESET}")
    return " ".join(result)
