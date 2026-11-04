"""Portable experiment entry points resolve the current checkout."""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "module",
    (
        "combo45_reference",
        "combo45_impact",
        "combo45_long",
        "combo45_impact_case",
    ),
)
def test_portable_help_needs_no_server_models_or_cuda(module):
    result = subprocess.run(
        [sys.executable, "-m", f"holoboxing.experiments.{module}", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=45,
        check=True,
    )
    for option in (
        "--assets",
        "--lower",
        "--source",
        "--output",
        "--no-video",
    ):
        assert option in result.stdout


def test_experiment_resolves_the_current_checkout():
    from holoboxing.experiments.combo45_reference import BASE

    assert BASE == ROOT
