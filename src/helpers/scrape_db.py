"""Database integration for scrape operations.

Provides functions to save scraped data to SQLite database
in addition to JSON files. This maintains backward compatibility
while adding database storage.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from db import init_db
from db.scraped_data import (
    create_scrape_session,
    add_scraped_pairs,
    get_pairs_for_session,
    list_scrape_sessions,
    export_session_to_json,
)


def save_scrape_to_db(
    source: str,
    pairs: List[Dict[str, str]],
    source_details: Optional[str] = None,
    dataset_format: str = "standard",
    metadata: Optional[Dict[str, Any]] = None,
) -> int:
    """Save scraped pairs to the database.
    
    Args:
        source: Source type (e.g., "4chan", "reddit", "stackexchange")
        pairs: List of dicts with 'input' and 'output' keys
        source_details: Additional details (e.g., board names, subreddit URL)
        dataset_format: Format type ("standard", "chatml")
        metadata: Optional metadata dictionary
        
    Returns:
        Session ID
    """
    init_db()
    
    # Create session
    session_id = create_scrape_session(
        source=source,
        source_details=source_details,
        dataset_format=dataset_format,
        metadata=metadata,
    )
    
    # Add pairs
    if pairs:
        add_scraped_pairs(session_id, pairs)
    
    return session_id


def save_chatml_to_db(
    source: str,
    conversations: List[Dict[str, Any]],
    source_details: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> int:
    """Save ChatML conversations to the database.
    
    Converts ChatML format to standard input/output pairs.
    
    Args:
        source: Source type
        conversations: List of ChatML conversation dicts
        source_details: Additional details
        metadata: Optional metadata
        
    Returns:
        Session ID
    """
    init_db()
    
    # Convert ChatML to pairs
    pairs = []
    for conv in conversations:
        messages = conv.get("messages", [])
        if not isinstance(messages, list) or len(messages) < 2:
            continue
        
        # Extract first user and assistant messages
        user_msg = None
        assistant_msg = None
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "")
            content = (msg.get("content") or "").strip()
            if not content:
                continue
            if role == "user" and user_msg is None:
                user_msg = content
            elif role == "assistant" and user_msg is not None and assistant_msg is None:
                assistant_msg = content
                break
        
        if user_msg and assistant_msg:
            pairs.append({"input": user_msg, "output": assistant_msg})
    
    # Create session
    session_id = create_scrape_session(
        source=source,
        source_details=source_details,
        dataset_format="chatml",
        metadata=metadata,
    )
    
    # Add pairs
    if pairs:
        add_scraped_pairs(session_id, pairs)
    
    return session_id


def load_pairs_from_db(
    session_id: Optional[int] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, str]]:
    """Load pairs from the database.
    
    Args:
        session_id: Optional session ID to filter by
        limit: Optional limit on number of pairs
        
    Returns:
        List of pair dictionaries
    """
    init_db()
    
    if session_id is not None:
        return get_pairs_for_session(session_id, limit=limit)
    
    # Get all pairs from all sessions
    from db.scraped_data import get_all_pairs
    return get_all_pairs(limit=limit)


def get_recent_sessions(limit: int = 10) -> List[Dict[str, Any]]:
    """Get recent scrape sessions.
    
    Args:
        limit: Maximum number of sessions to return
        
    Returns:
        List of session dictionaries
    """
    init_db()
    return list_scrape_sessions(limit=limit)


def export_to_json(session_id: int, output_path: str) -> int:
    """Export a session to JSON file.
    
    Args:
        session_id: Session ID to export
        output_path: Path to write JSON file
        
    Returns:
        Number of pairs exported
    """
    init_db()
    return export_session_to_json(session_id, output_path)
