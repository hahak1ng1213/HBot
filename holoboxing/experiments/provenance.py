"""Record code identity without modifying the frozen teacher implementation."""

import subprocess
from pathlib import Path


def code_identity():
    """Return the revision and dirty flag, or explicitly mark an archive."""
    root = Path(__file__).resolve().parents[2]
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, timeout=5
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True, timeout=5
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return {"revision": None, "dirty": None, "kind": "source_archive"}
    return {"revision": revision, "dirty": bool(status), "kind": "git"}
