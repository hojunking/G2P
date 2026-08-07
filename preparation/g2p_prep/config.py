"""YAML configuration loading for the preparation stage."""

import os

import yaml

REQUIRED_KEYS = ("scene_root", "gs_root")


def load_config(path, overrides=None):
    """Load a preparation-stage YAML config.

    Args:
        path: path to a YAML file under ``preparation/configs/``.
        overrides: mapping of key -> value applied after loading; ``None``
            values are ignored so unset CLI flags do not clobber the file.

    Returns:
        dict with ``scene_root``, ``gs_root``, ``meta_root`` and any extra keys.
    """
    with open(path) as f:
        config = yaml.safe_load(f) or {}

    for key, value in (overrides or {}).items():
        if value is not None:
            config[key] = value

    missing = [k for k in REQUIRED_KEYS if not config.get(k)]
    if missing:
        raise ValueError(
            f"{path}: missing required key(s) {missing}. Set them in the YAML "
            f"file or pass the matching --{missing[0].replace('_', '-')} flag."
        )

    config.setdefault(
        "meta_root",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(path))), "meta"),
    )
    return config
