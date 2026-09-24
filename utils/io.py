"""Atomic file I/O helpers for JSON and CSV result serialization."""

import json
from pathlib import Path
import tempfile
from typing import Any, Dict
import numpy as np
import pandas as pd


def _to_serializable(obj: Any) -> Any:
    """Recursively convert NumPy and Path types to standard JSON-compatible Python types."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    return obj


def save_json(data: Dict[str, Any], path: Path, indent: int = 2) -> None:
    """Save dictionary to JSON atomically using a temporary file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = _to_serializable(data)

    with tempfile.NamedTemporaryFile("w", dir=str(path.parent), delete=False, encoding="utf-8") as f:
        json.dump(serializable, f, indent=indent)
        temp_name = f.name

    Path(temp_name).replace(path)


def load_json(path: Path) -> Dict[str, Any]:
    """Load JSON file from disk."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    """Save DataFrame to CSV atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=str(path.parent), delete=False, encoding="utf-8", newline="") as f:
        df.to_csv(f, index=False)
        temp_name = f.name
    Path(temp_name).replace(path)
