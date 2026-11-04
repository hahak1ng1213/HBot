"""Check the curated boxing additions without model downloads or a GPU."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = (
    "solo_skill_physics",
    "solo_skill_reference",
    "solo_skill_env",
    "solo_skill_training",
    "solo_skill_evaluate",
    "solo_waypoint",
    "turn_reference",
    "boxing_contact_geometry",
    "boxing_impulse_metrics",
    "boxing_experiment_cli",
    "boxing_cpu_physics",
    "boxing_contact_diagnostics",
    "boxing_fk_cuda_graph",
)
LINT = (
    "holoboxing/training/skills",
    "holoboxing/experiments",
    "holoboxing/data/augment_punch.py",
    "holoboxing/evaluation/runtime/contact_geometry.py",
    "holoboxing/evaluation/runtime/scene.py",
    "holoboxing/training/envs/official_referee.py",
    "tools/check_boxing_baseline.py",
    *tuple(f"tests/test_{name}.py" for name in TESTS[:10]),
)


def main():
    manifest = ROOT / "holoboxing/experiments/frozen_skill_sources.json"
    for relative, expected in json.loads(manifest.read_text()).items():
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Frozen teacher source changed: {relative}")
    commands = (
        [sys.executable, "-m", "ruff", "check", *LINT],
        [sys.executable, "-m", "ruff", "format", "--check", *LINT],
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            *(f"tests/test_{name}.py" for name in TESTS),
            *sys.argv[1:],
        ],
    )
    for command in commands:
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
