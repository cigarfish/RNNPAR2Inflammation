"""Load config.yaml and resolve all paths against the project root."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config.yaml"


class Config(dict):
    """dict with attribute access; cfg.paths holds resolved Path objects."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(key) from e


def load_config(path=DEFAULT_CONFIG):
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    cfg = Config(raw)
    cfg["root"] = ROOT
    cfg["paths"] = Config({k: (ROOT / v).resolve() for k, v in raw["paths"].items()})
    cfg["external_contrasts"] = Config(raw.get("external_contrasts", {}))
    return cfg


def external_file(cfg, stem):
    """Path of one MEK/ERK contrast table in data_external."""
    return cfg.paths.external_dir / f"{stem}.tsv"
