"""Training runtime setup; no simulator at import."""

import json
import subprocess
from pathlib import Path

from holoboxing.training.resources import resolve_resources
from holoboxing.training.tasks.combat.config.sandbag import sha256


def prepare(args):
    _, _, resources = resolve_resources(args.assets, args.lower)
    if args.output.exists() and (
        not args.output.is_dir() or any(args.output.iterdir())
    ):
        raise ValueError("Use a new empty output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    package = Path(__file__).resolve().parents[1]
    identity = {
        "resources": {
            key: value["sha256"] for key, value in resources.items()
        },
        "code": {
            str(path.relative_to(package)): sha256(path)
            for path in sorted(
                [*package.rglob("*.py"), *package.glob("assets/*.json")]
            )
            if "tests" not in path.parts
        },
    }
    root = package.parents[1]
    # The shared referee, observations and kinematics are part of a run's code,
    # even though they live outside training/.
    identity["shared_code"] = {
        path.relative_to(root).as_posix(): sha256(path)
        for directory in (
            package.parent / "evaluation",
            package.parent / "data",
            root / "holomotion/src/motion_tracking",
        )
        for path in sorted(directory.rglob("*.py"))
    }
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(package), "rev-parse", "HEAD"], text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        revision = None
    write_json(
        args.output / "inputs.json",
        {
            "resources": resources,
            "identity": identity,
            "holomotion_revision": revision,
            "arguments": vars(args),
        },
    )
    return identity


def write_json(path, record):
    Path(path).write_text(
        json.dumps(record, default=str, indent=2, allow_nan=False) + "\n"
    )
