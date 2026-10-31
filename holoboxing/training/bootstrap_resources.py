"""Download the pinned public boxing inventory and verify every resource."""

import argparse
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen

from holoboxing.evaluation.models import _download_file
from holoboxing.evaluation.resources import ResourceTree

REVISION = "795e5e1c1d3b6b15cc1cea9e7752742715724826"
ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
BASE = f"{ENDPOINT}/datasets/HorizonRobotics/HoloMotion_assets/resolve/{REVISION}/holoboxing/"


def bootstrap(directory):
    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "HoloBoxing/1.0"}
    with urlopen(Request(BASE + "resources.json", headers=headers), timeout=60) as response:
        raw = response.read()
    inventory = json.loads(raw)
    for entry in inventory["files"]:
        name = Path(entry["path"])
        path = (root / name).resolve()
        if name.is_absolute() or not path.is_relative_to(root):
            raise ValueError("Unsafe resource path")
        path.parent.mkdir(parents=True, exist_ok=True)
        from holoboxing.evaluation.runtime.model_contract import sha256

        if path.exists():
            if (
                path.stat().st_size == entry["bytes"]
                and sha256(path) == entry["sha256"]
            ):
                continue
            raise ValueError(f"Existing resource differs: {path}")
        _download_file(path, BASE + name.as_posix(), entry, retries=2)
    manifest = root / "resources.json"
    if manifest.exists() and manifest.read_bytes() != raw:
        raise ValueError("Existing manifest differs")
    manifest.write_bytes(raw)
    replay = root / "manifest.json"
    with urlopen(Request(BASE + "manifest.json", headers=headers), timeout=60) as response:
        replay_raw = response.read()
    if replay.exists() and replay.read_bytes() != replay_raw:
        raise ValueError("Existing replay manifest differs")
    replay.write_bytes(replay_raw)
    report = ResourceTree(root).verify()
    receipt = dict(
        asset_revision=REVISION,
        manifest_sha256=sha256(manifest),
        verification=report,
    )
    (root / "download-receipt.json").write_text(json.dumps(receipt, indent=2))
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assets", required=True)
    args = p.parse_args()
    print(json.dumps(bootstrap(args.assets), indent=2))


if __name__ == "__main__":
    main()
