"""File-based feature persistence with metadata and versioning.

Each save writes a parquet file plus a JSON metadata sidecar describing the
generation time, feature list, date range and pairs. Filenames carry a
timestamp so multiple versions coexist; only the most recent versions are
retained. Writes are atomic (write to a temp file then rename) so a crash
mid-write cannot corrupt an existing version.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path

import pandas as pd

from config.logging_config import get_logger

logger = get_logger()

_DEFAULT_DIR = Path("data/processed/features")
_KEEP_VERSIONS = 5
_TIMESTAMP_FORMAT = "%Y%m%dT%H%M%S%f"


def _atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame to parquet atomically via a temp file and rename.

    Args:
        frame: The DataFrame to persist.
        path: Destination parquet path.

    Returns:
        None.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(tmp)
    os.replace(tmp, path)


def _atomic_write_json(payload: dict, path: Path) -> None:
    """Write a JSON sidecar atomically.

    Args:
        payload: JSON-serialisable metadata.
        path: Destination JSON path.

    Returns:
        None.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    os.replace(tmp, path)


def _flatten_feature_names(columns: pd.Index) -> list[str]:
    """Render column labels (possibly a MultiIndex) as flat strings.

    Args:
        columns: The columns of the feature frame.

    Returns:
        A list of string labels.
    """
    if isinstance(columns, pd.MultiIndex):
        return ["::".join(str(level) for level in tup) for tup in columns]
    return [str(c) for c in columns]


def save_features(
    frame: pd.DataFrame,
    name: str = "features",
    store_dir: Path = _DEFAULT_DIR,
    keep_versions: int = _KEEP_VERSIONS,
) -> Path:
    """Persist a feature frame with a metadata sidecar and prune old versions.

    Args:
        frame: Feature DataFrame to store; index must be datetime-like.
        name: Logical dataset name used as the filename stem.
        store_dir: Directory holding versions of this dataset.
        keep_versions: Number of most recent versions to retain.

    Returns:
        The path of the parquet file that was written.

    Raises:
        ValueError: If the frame is empty.
    """
    if frame.empty:
        raise ValueError("Refusing to save an empty feature frame.")

    store_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime(_TIMESTAMP_FORMAT)
    parquet_path = store_dir / f"{name}_{stamp}.parquet"
    meta_path = store_dir / f"{name}_{stamp}.json"

    pairs: list[str] = []
    if isinstance(frame.columns, pd.MultiIndex):
        pairs = sorted({str(p) for p in frame.columns.get_level_values(0)})

    metadata = {
        "name": name,
        "generated_at": dt.datetime.now().isoformat(),
        "n_rows": int(len(frame)),
        "n_features": int(frame.shape[1]),
        "feature_list": _flatten_feature_names(frame.columns),
        "pairs": pairs,
        "date_start": str(frame.index.min()),
        "date_end": str(frame.index.max()),
    }

    _atomic_write_parquet(frame, parquet_path)
    _atomic_write_json(metadata, meta_path)
    logger.info("Saved features to {} ({} rows, {} cols)", parquet_path, len(frame), frame.shape[1])

    _prune_old_versions(name, store_dir, keep_versions)
    return parquet_path


def _prune_old_versions(name: str, store_dir: Path, keep_versions: int) -> None:
    """Delete all but the most recent keep_versions of a dataset.

    Args:
        name: Dataset stem.
        store_dir: Directory containing the versions.
        keep_versions: Number of versions to retain.

    Returns:
        None.
    """
    versions = sorted(store_dir.glob(f"{name}_*.parquet"))
    excess = versions[:-keep_versions] if keep_versions > 0 else []
    for old in excess:
        old.unlink(missing_ok=True)
        old.with_suffix(".json").unlink(missing_ok=True)
        logger.debug("Pruned old feature version {}", old)


def latest_version_path(name: str = "features", store_dir: Path = _DEFAULT_DIR) -> Path | None:
    """Return the path of the most recent parquet version, if any.

    Args:
        name: Dataset stem.
        store_dir: Directory containing the versions.

    Returns:
        The newest parquet path, or None if no version exists.
    """
    versions = sorted(store_dir.glob(f"{name}_*.parquet"))
    return versions[-1] if versions else None


def load_features(name: str = "features", store_dir: Path = _DEFAULT_DIR) -> pd.DataFrame:
    """Load the most recent version of a feature dataset.

    Args:
        name: Dataset stem.
        store_dir: Directory containing the versions.

    Returns:
        The most recent feature DataFrame.

    Raises:
        FileNotFoundError: If no version exists for the given name.
    """
    path = latest_version_path(name, store_dir)
    if path is None:
        raise FileNotFoundError(f"No saved feature version found for '{name}' in {store_dir}.")
    logger.info("Loading features from {}", path)
    return pd.read_parquet(path)
