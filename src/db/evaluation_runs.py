"""Database functions for evaluation runs."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .core import get_connection, init_db


def save_evaluation_run(
    *,
    base_model: str,
    benchmark: str,
    metrics: Dict[str, Any],
    training_run_id: Optional[int] = None,
    adapter_path: Optional[str] = None,
    num_samples: Optional[int] = None,
    batch_size: Optional[int] = None,
    is_base_eval: bool = False,
    db_path: Optional[str] = None,
) -> int:
    """Save an evaluation run to the database.
    
    Args:
        base_model: The base model used for evaluation
        benchmark: The benchmark name (e.g., 'hellaswag', 'truthfulqa_mc2')
        metrics: Dictionary of metric results
        training_run_id: Optional ID of the associated training run
        adapter_path: Path to the adapter used (if any)
        num_samples: Number of samples evaluated
        batch_size: Batch size used
        is_base_eval: True if this was a base model evaluation (no adapter)
        db_path: Optional database path override
        
    Returns:
        The ID of the newly created evaluation run
    """
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()
    
    cursor.execute(
        """
        INSERT INTO evaluation_runs 
        (training_run_id, base_model, adapter_path, benchmark, num_samples, 
         batch_size, metrics_json, is_base_eval)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            training_run_id,
            base_model,
            adapter_path,
            benchmark,
            num_samples,
            batch_size,
            json.dumps(metrics),
            1 if is_base_eval else 0,
        ),
    )
    conn.commit()
    return cursor.lastrowid


def get_evaluation_run(eval_id: int, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Get a single evaluation run by ID.
    
    Args:
        eval_id: The evaluation run ID
        db_path: Optional database path override
        
    Returns:
        Dictionary with evaluation run data, or None if not found
    """
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM evaluation_runs WHERE id = ?", (eval_id,))
    row = cursor.fetchone()
    
    if row is None:
        return None
    
    return _row_to_dict(row)


def get_evaluation_runs_for_training(
    training_run_id: int,
    db_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Get all evaluation runs for a specific training run.
    
    Args:
        training_run_id: The training run ID
        db_path: Optional database path override
        
    Returns:
        List of evaluation run dictionaries
    """
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()
    
    cursor.execute(
        """
        SELECT * FROM evaluation_runs 
        WHERE training_run_id = ?
        ORDER BY created_at DESC
        """,
        (training_run_id,),
    )
    
    return [_row_to_dict(row) for row in cursor.fetchall()]


def get_recent_evaluation_runs(
    limit: int = 20,
    benchmark: Optional[str] = None,
    db_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Get recent evaluation runs, optionally filtered by benchmark.
    
    Args:
        limit: Maximum number of runs to return
        benchmark: Optional benchmark name to filter by
        db_path: Optional database path override
        
    Returns:
        List of evaluation run dictionaries
    """
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()
    
    if benchmark:
        cursor.execute(
            """
            SELECT * FROM evaluation_runs 
            WHERE benchmark = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (benchmark, limit),
        )
    else:
        cursor.execute(
            """
            SELECT * FROM evaluation_runs 
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        )
    
    return [_row_to_dict(row) for row in cursor.fetchall()]


def get_evaluation_history_for_model(
    base_model: str,
    adapter_path: Optional[str] = None,
    db_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Get evaluation history for a specific model/adapter combination.
    
    Args:
        base_model: The base model name
        adapter_path: Optional adapter path (None for base model only)
        db_path: Optional database path override
        
    Returns:
        List of evaluation run dictionaries
    """
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()
    
    if adapter_path:
        cursor.execute(
            """
            SELECT * FROM evaluation_runs 
            WHERE base_model = ? AND adapter_path = ?
            ORDER BY created_at DESC
            """,
            (base_model, adapter_path),
        )
    else:
        cursor.execute(
            """
            SELECT * FROM evaluation_runs 
            WHERE base_model = ? AND adapter_path IS NULL
            ORDER BY created_at DESC
            """,
            (base_model,),
        )
    
    return [_row_to_dict(row) for row in cursor.fetchall()]


def delete_evaluation_run(eval_id: int, db_path: Optional[str] = None) -> bool:
    """Delete an evaluation run.
    
    Args:
        eval_id: The evaluation run ID to delete
        db_path: Optional database path override
        
    Returns:
        True if deleted, False if not found
    """
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()
    
    cursor.execute("DELETE FROM evaluation_runs WHERE id = ?", (eval_id,))
    conn.commit()
    return cursor.rowcount > 0


def _row_to_dict(row) -> Dict[str, Any]:
    """Convert a database row to a dictionary with parsed JSON."""
    result = dict(row)
    
    # Parse metrics JSON
    if result.get("metrics_json"):
        try:
            result["metrics"] = json.loads(result["metrics_json"])
        except json.JSONDecodeError:
            result["metrics"] = {}
    else:
        result["metrics"] = {}
    
    # Convert is_base_eval to boolean
    result["is_base_eval"] = bool(result.get("is_base_eval", 0))
    
    return result
