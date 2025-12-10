"""Training runs database operations for managed storage."""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from typing import Any, Dict, List, Optional

from db.core import get_connection, get_db_path

# Default managed storage directory (relative to project root)
_MANAGED_STORAGE_DIR = "training_outputs"


def get_managed_storage_root() -> str:
    """Get the root directory for managed training storage."""
    db_path = get_db_path()
    project_root = os.path.dirname(os.path.abspath(db_path))
    storage_root = os.path.join(project_root, _MANAGED_STORAGE_DIR)
    os.makedirs(storage_root, exist_ok=True)
    return storage_root


def create_training_run(
    name: str,
    base_model: str,
    dataset_source: str,
    dataset_id: str,
    hp: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a new training run with managed storage.

    Returns the created run record with storage_path set up.
    """
    conn = get_connection()
    cursor = conn.cursor()

    # Create unique storage directory for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    storage_name = f"{safe_name}_{timestamp}"
    storage_root = get_managed_storage_root()
    storage_path = os.path.join(storage_root, storage_name)

    # Create the directory structure
    os.makedirs(storage_path, exist_ok=True)
    os.makedirs(os.path.join(storage_path, "outputs"), exist_ok=True)
    os.makedirs(os.path.join(storage_path, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(storage_path, "logs"), exist_ok=True)

    hp_json = json.dumps(hp) if hp else None
    metadata_json = json.dumps(metadata) if metadata else None

    cursor.execute(
        """
        INSERT INTO training_runs 
        (name, status, base_model, dataset_source, dataset_id, storage_path, hp_json, metadata_json)
        VALUES (?, 'pending', ?, ?, ?, ?, ?, ?)
    """,
        (name, base_model, dataset_source, dataset_id, storage_path, hp_json, metadata_json),
    )

    run_id = cursor.lastrowid
    conn.commit()

    return get_training_run(run_id)


def get_training_run(run_id: int) -> Optional[Dict[str, Any]]:
    """Get a training run by ID."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM training_runs WHERE id = ?", (run_id,))
    row = cursor.fetchone()
    if not row:
        return None
    return _row_to_dict(row)


def list_training_runs(
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """List training runs, optionally filtered by status."""
    conn = get_connection()
    cursor = conn.cursor()

    if status:
        cursor.execute(
            """
            SELECT * FROM training_runs 
            WHERE status = ?
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
        """,
            (status, limit, offset),
        )
    else:
        cursor.execute(
            """
            SELECT * FROM training_runs 
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
        """,
            (limit, offset),
        )

    return [_row_to_dict(row) for row in cursor.fetchall()]


def update_training_run(
    run_id: int,
    status: Optional[str] = None,
    output_dir: Optional[str] = None,
    adapter_path: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    logs: Optional[List[str]] = None,
    started_at: Optional[str] = None,
    completed_at: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Update a training run record."""
    conn = get_connection()
    cursor = conn.cursor()

    updates = []
    params = []

    if status is not None:
        updates.append("status = ?")
        params.append(status)
    if output_dir is not None:
        updates.append("output_dir = ?")
        params.append(output_dir)
    if adapter_path is not None:
        updates.append("adapter_path = ?")
        params.append(adapter_path)
    if checkpoint_path is not None:
        updates.append("checkpoint_path = ?")
        params.append(checkpoint_path)
    if logs is not None:
        updates.append("logs_json = ?")
        params.append(json.dumps(logs))
    if started_at is not None:
        updates.append("started_at = ?")
        params.append(started_at)
    if completed_at is not None:
        updates.append("completed_at = ?")
        params.append(completed_at)
    if metadata is not None:
        updates.append("metadata_json = ?")
        params.append(json.dumps(metadata))

    if not updates:
        return get_training_run(run_id)

    updates.append("updated_at = datetime('now')")
    params.append(run_id)

    cursor.execute(
        f"""
        UPDATE training_runs 
        SET {", ".join(updates)}
        WHERE id = ?
    """,
        params,
    )
    conn.commit()

    return get_training_run(run_id)


def delete_training_run(run_id: int, delete_files: bool = True) -> bool:
    """Delete a training run and optionally its files."""
    run = get_training_run(run_id)
    if not run:
        return False

    # Delete storage directory if requested
    if delete_files and run.get("storage_path"):
        storage_path = run["storage_path"]
        if os.path.isdir(storage_path):
            try:
                shutil.rmtree(storage_path)
            except Exception:
                pass

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM training_runs WHERE id = ?", (run_id,))
    conn.commit()

    return cursor.rowcount > 0


def get_run_storage_paths(run_id: int) -> Optional[Dict[str, str]]:
    """Get all storage paths for a training run."""
    run = get_training_run(run_id)
    if not run or not run.get("storage_path"):
        return None

    storage_path = run["storage_path"]
    return {
        "root": storage_path,
        "outputs": os.path.join(storage_path, "outputs"),
        "checkpoints": os.path.join(storage_path, "checkpoints"),
        "logs": os.path.join(storage_path, "logs"),
        "dataset": os.path.join(storage_path, "dataset.json"),
    }


def get_latest_run() -> Optional[Dict[str, Any]]:
    """Get the most recent training run."""
    runs = list_training_runs(limit=1)
    return runs[0] if runs else None


def get_runs_by_model(base_model: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Get training runs for a specific base model."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT * FROM training_runs 
        WHERE base_model = ?
        ORDER BY created_at DESC
        LIMIT ?
    """,
        (base_model, limit),
    )
    return [_row_to_dict(row) for row in cursor.fetchall()]


def _row_to_dict(row) -> Dict[str, Any]:
    """Convert a database row to a dictionary."""
    d = dict(row)
    # Parse JSON fields
    if d.get("hp_json"):
        try:
            d["hp"] = json.loads(d["hp_json"])
        except Exception:
            d["hp"] = {}
    else:
        d["hp"] = {}
    if d.get("logs_json"):
        try:
            d["logs"] = json.loads(d["logs_json"])
        except Exception:
            d["logs"] = []
    else:
        d["logs"] = []
    if d.get("metadata_json"):
        try:
            d["metadata"] = json.loads(d["metadata_json"])
        except Exception:
            d["metadata"] = {}
    else:
        d["metadata"] = {}
    return d
