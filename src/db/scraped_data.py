"""Scraped data storage using SQLite.

Replaces JSON files for scraped training data with database storage.
Supports sessions (scrape runs) and individual pairs.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .core import get_connection, init_db


def create_scrape_session(
    source: str,
    source_details: Optional[str] = None,
    dataset_format: str = "standard",
    metadata: Optional[Dict[str, Any]] = None,
    db_path: Optional[str] = None,
) -> int:
    """Create a new scrape session.

    Args:
        source: Source type (e.g., "4chan", "reddit", "stackexchange")
        source_details: Additional details (e.g., board names, subreddit URL)
        dataset_format: Format type ("standard", "chatml", etc.)
        metadata: Optional metadata dictionary
        db_path: Optional database path

    Returns:
        Session ID
    """
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()

    metadata_json = json.dumps(metadata, ensure_ascii=False) if metadata else None

    cursor.execute(
        """
        INSERT INTO scrape_sessions (source, source_details, dataset_format, metadata_json)
        VALUES (?, ?, ?, ?)
    """,
        (source, source_details, dataset_format, metadata_json),
    )

    conn.commit()
    return cursor.lastrowid


def add_scraped_pairs(session_id: int, pairs: List[Dict[str, str]], db_path: Optional[str] = None) -> int:
    """Add scraped pairs to a session.

    Args:
        session_id: Session ID to add pairs to
        pairs: List of dicts with 'input' and 'output' keys
        db_path: Optional database path

    Returns:
        Number of pairs added
    """
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()

    added = 0
    for pair in pairs:
        input_text = pair.get("input", "")
        output_text = pair.get("output", "")

        if not input_text or not output_text:
            continue

        source_url = pair.get("source_url")
        metadata = {k: v for k, v in pair.items() if k not in ("input", "output", "source_url")}
        metadata_json = json.dumps(metadata, ensure_ascii=False) if metadata else None

        cursor.execute(
            """
            INSERT INTO scraped_pairs (session_id, input_text, output_text, source_url, metadata_json)
            VALUES (?, ?, ?, ?, ?)
        """,
            (session_id, input_text, output_text, source_url, metadata_json),
        )
        added += 1

    # Update pair count on session
    cursor.execute(
        """
        UPDATE scrape_sessions 
        SET pair_count = (SELECT COUNT(*) FROM scraped_pairs WHERE session_id = ?)
        WHERE id = ?
    """,
        (session_id, session_id),
    )

    conn.commit()
    return added


def get_scrape_session(session_id: int, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Get a scrape session by ID.

    Args:
        session_id: Session ID
        db_path: Optional database path

    Returns:
        Session dictionary or None
    """
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT id, source, source_details, dataset_format, pair_count, created_at, metadata_json
        FROM scrape_sessions WHERE id = ?
    """,
        (session_id,),
    )

    row = cursor.fetchone()
    if row is None:
        return None

    result = {
        "id": row["id"],
        "source": row["source"],
        "source_details": row["source_details"],
        "dataset_format": row["dataset_format"],
        "pair_count": row["pair_count"],
        "created_at": row["created_at"],
    }

    if row["metadata_json"]:
        try:
            result["metadata"] = json.loads(row["metadata_json"])
        except (json.JSONDecodeError, TypeError):
            pass

    return result


def list_scrape_sessions(
    source: Optional[str] = None, limit: int = 100, db_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """List scrape sessions.

    Args:
        source: Optional filter by source type
        limit: Maximum number of sessions to return
        db_path: Optional database path

    Returns:
        List of session dictionaries
    """
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()

    if source:
        cursor.execute(
            """
            SELECT id, source, source_details, dataset_format, pair_count, created_at
            FROM scrape_sessions
            WHERE source = ?
            ORDER BY created_at DESC
            LIMIT ?
        """,
            (source, limit),
        )
    else:
        cursor.execute(
            """
            SELECT id, source, source_details, dataset_format, pair_count, created_at
            FROM scrape_sessions
            ORDER BY created_at DESC
            LIMIT ?
        """,
            (limit,),
        )

    return [dict(row) for row in cursor.fetchall()]


def get_pairs_for_session(
    session_id: int, limit: Optional[int] = None, offset: int = 0, db_path: Optional[str] = None
) -> List[Dict[str, str]]:
    """Get pairs for a scrape session.

    Args:
        session_id: Session ID
        limit: Optional limit on number of pairs
        offset: Offset for pagination
        db_path: Optional database path

    Returns:
        List of pair dictionaries with 'input' and 'output' keys
    """
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()

    if limit:
        cursor.execute(
            """
            SELECT input_text, output_text, source_url, metadata_json
            FROM scraped_pairs
            WHERE session_id = ?
            ORDER BY id
            LIMIT ? OFFSET ?
        """,
            (session_id, limit, offset),
        )
    else:
        cursor.execute(
            """
            SELECT input_text, output_text, source_url, metadata_json
            FROM scraped_pairs
            WHERE session_id = ?
            ORDER BY id
        """,
            (session_id,),
        )

    pairs = []
    for row in cursor.fetchall():
        pair = {
            "input": row["input_text"],
            "output": row["output_text"],
        }
        if row["source_url"]:
            pair["source_url"] = row["source_url"]
        if row["metadata_json"]:
            try:
                metadata = json.loads(row["metadata_json"])
                pair.update(metadata)
            except (json.JSONDecodeError, TypeError):
                pass
        pairs.append(pair)

    return pairs


def export_session_to_json(session_id: int, output_path: str, db_path: Optional[str] = None) -> int:
    """Export a scrape session to a JSON file.

    Args:
        session_id: Session ID to export
        output_path: Path to write JSON file
        db_path: Optional database path

    Returns:
        Number of pairs exported
    """
    pairs = get_pairs_for_session(session_id, db_path=db_path)

    # Convert to standard format (just input/output)
    export_pairs = [{"input": p["input"], "output": p["output"]} for p in pairs]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(export_pairs, f, ensure_ascii=False, indent=2)

    return len(export_pairs)


def delete_scrape_session(session_id: int, db_path: Optional[str] = None) -> bool:
    """Delete a scrape session and all its pairs.

    Args:
        session_id: Session ID to delete
        db_path: Optional database path

    Returns:
        True if session was deleted, False if it didn't exist
    """
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()

    # Pairs are deleted automatically via ON DELETE CASCADE
    cursor.execute("DELETE FROM scrape_sessions WHERE id = ?", (session_id,))
    conn.commit()

    return cursor.rowcount > 0


def get_all_pairs(limit: Optional[int] = None, db_path: Optional[str] = None) -> List[Dict[str, str]]:
    """Get all pairs across all sessions.

    Args:
        limit: Optional limit on number of pairs
        db_path: Optional database path

    Returns:
        List of pair dictionaries
    """
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()

    if limit:
        cursor.execute(
            """
            SELECT input_text, output_text
            FROM scraped_pairs
            ORDER BY id
            LIMIT ?
        """,
            (limit,),
        )
    else:
        cursor.execute("""
            SELECT input_text, output_text
            FROM scraped_pairs
            ORDER BY id
        """)

    return [{"input": row["input_text"], "output": row["output_text"]} for row in cursor.fetchall()]


def get_total_pair_count(db_path: Optional[str] = None) -> int:
    """Get total number of pairs across all sessions.

    Returns:
        Total pair count
    """
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) as count FROM scraped_pairs")
    row = cursor.fetchone()

    return row["count"] if row else 0
