"""Vector store interface — Phase 2 stub.

All functions are no-ops until Phase 2 (LanceDB local vector layer) is implemented.
See PLAN.md § Phase 2 for the full specification.
"""

import pathlib


def on_doc_saved(path: pathlib.Path) -> None:
    """Called after a transcript document is saved to ~/transcript/.

    Phase 2: extract the '## Summary' section from the markdown, embed it,
    and upsert the vector + metadata into LanceDB.
    """
    pass  # TODO: Phase 2
