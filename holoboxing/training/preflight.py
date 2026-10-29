"""Write a resource/runtime readiness receipt without running simulation."""

import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys


def inspect(assets, lower):
    from holoboxing.evaluation.resources import ResourceTree
    from holoboxing.evaluation.models import verify_bundle, frozen_manifest

    report = dict(
        python=sys.executable,
        resources={},
        dependencies={},
        training_run=False,
        boxing_quality_verified=False,
    )
    for name, check in [
        ("assets", lambda: ResourceTree(assets).verify()),
        ("lower", lambda: str(verify_bundle(lower))),
    ]:
        try:
            report["resources"][name] = dict(status="PASS", result=check())
        except Exception as error:
            report["resources"][name] = dict(
                status="BLOCKED", error=str(error)
            )
    report["frozen_model"] = frozen_manifest()
    for name in (
        "torch",
        "numpy",
        "onnx",
        "onnxruntime",
        "mujoco",
        "mujoco_warp",
        "warp",
    ):
        report["dependencies"][name] = (
            importlib.util.find_spec(name) is not None
        )
    report["nvidia_smi"] = shutil.which("nvidia-smi")
    if report["nvidia_smi"]:
        probe = subprocess.run(
            [report["nvidia_smi"], "-L"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        report["gpu_probe"] = dict(
            exit_code=probe.returncode, output=probe.stdout, error=probe.stderr
        )
    report["ready_for_runtime_probe"] = (
        all(x["status"] == "PASS" for x in report["resources"].values())
        and all(report["dependencies"].values())
        and report.get("gpu_probe", {}).get("exit_code") == 0
    )
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assets", required=True)
    p.add_argument("--lower", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report = inspect(args.assets, args.lower)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    return 0 if report["ready_for_runtime_probe"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
