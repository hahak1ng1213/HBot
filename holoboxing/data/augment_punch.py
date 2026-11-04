"""Compatibility catalog location for the preserved solo-skill source.

The experimental punch augmentation pipeline is deliberately not included.
This module only preserves the CATALOG import used by frozen teacher code.
"""

from pathlib import Path

CATALOG = Path(__file__).parent / "profiles" / "robostriker.json"
