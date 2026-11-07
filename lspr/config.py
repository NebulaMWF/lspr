"""YAML config files as argparse defaults (command-line flags still take precedence)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import yaml


def parse_with_config(parser: argparse.ArgumentParser, argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse ``--config file.yaml`` first, use its keys (dashes or underscores) as defaults."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=None)
    known, _ = pre.parse_known_args(argv)
    parser.add_argument("--config", type=Path, default=None, help="YAML file with default arguments")
    if known.config is not None:
        cfg = yaml.safe_load(Path(known.config).read_text()) or {}
        actions = {a.dest: a for a in parser._actions}
        defaults = {}
        for key, value in cfg.items():
            dest = str(key).replace("-", "_")
            if dest not in actions:
                raise ValueError(f"{known.config}: unknown option {key!r}")
            # argparse converts string defaults itself, but not list elements.
            convert = actions[dest].type
            if isinstance(value, list) and callable(convert):
                value = [convert(v) for v in value]
            defaults[dest] = value
        parser.set_defaults(**defaults)
    return parser.parse_args(argv)
