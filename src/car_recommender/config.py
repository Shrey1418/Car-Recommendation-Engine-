from functools import cache
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]

@cache
def load_config(path=None):
    config_path = Path(path) if path else PROJECT_ROOT / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def resolve_path(relative):
    return (PROJECT_ROOT / relative).resolve()