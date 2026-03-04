"""voice-index: backfill the vector index for existing transcripts.

Usage:
    voice-index                  # indexes all *.md files in ~/transcript/
    voice-index --dir /some/path # indexes a custom directory
"""

import argparse
import pathlib
import sys

from voice_transcribe import vector
from voice_transcribe.config import TRANSCRIPT_DIR


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill the vector index for existing transcript files."
    )
    parser.add_argument(
        "--dir",
        type=pathlib.Path,
        default=TRANSCRIPT_DIR,
        help=f"Directory to scan for .md files (default: {TRANSCRIPT_DIR})",
    )
    args = parser.parse_args()

    files = sorted(args.dir.glob("*.md"))
    if not files:
        print(f"No .md files found in {args.dir}")
        sys.exit(0)

    print(f"Indexing {len(files)} file(s) from {args.dir} ...")
    ok = 0
    failed = 0
    for i, f in enumerate(files, 1):
        try:
            vector.on_doc_saved(f)
            print(f"  [{i}/{len(files)}] ok   {f.name}")
            ok += 1
        except Exception as e:
            print(f"  [{i}/{len(files)}] FAIL {f.name}: {e}")
            failed += 1

    print(f"\nDone: {ok} indexed, {failed} failed.")


if __name__ == "__main__":
    main()
