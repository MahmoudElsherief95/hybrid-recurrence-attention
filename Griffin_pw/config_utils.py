import yaml
from typing import Any, Dict

def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML config file and return as a dictionary."""
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config
