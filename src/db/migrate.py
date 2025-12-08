"""Migration utilities for importing existing JSON data into SQLite.

Handles one-time migration from:
- ff_settings.json -> settings table
- saved_configs/*.json -> training_configs table
- scraped_training_data.json -> scraped_pairs table
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from .core import get_connection, init_db, get_db_path
from .settings import set_setting
from .training_configs import save_training_config
from .scraped_data import create_scrape_session, add_scraped_pairs


def migrate_from_json(
    project_root: Optional[str] = None, db_path: Optional[str] = None, delete_after: bool = False
) -> Dict[str, Any]:
    """Migrate all JSON data to SQLite.

    Args:
        project_root: Project root directory (contains ff_settings.json, saved_configs/, etc.)
        db_path: Optional database path
        delete_after: If True, delete JSON files after successful migration

    Returns:
        Dictionary with migration results
    """
    if project_root is None:
        # Default: parent of src/
        src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        project_root = os.path.dirname(src_dir)

    if db_path is None:
        db_path = get_db_path(project_root)

    # Initialize database
    init_db(db_path)

    results = {
        "settings": {"migrated": False, "error": None},
        "training_configs": {"migrated": 0, "errors": []},
        "scraped_data": {"migrated": 0, "sessions": 0, "errors": []},
    }

    # Migrate settings
    settings_result = _migrate_settings(project_root, db_path)
    results["settings"] = settings_result

    # Migrate training configs
    configs_result = _migrate_training_configs(project_root, db_path)
    results["training_configs"] = configs_result

    # Migrate scraped data
    scraped_result = _migrate_scraped_data(project_root, db_path)
    results["scraped_data"] = scraped_result

    # Mark migration as complete
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO app_state (key, value)
        VALUES ('json_migration_complete', 'true')
    """)
    conn.commit()

    # Optionally delete JSON files
    if delete_after:
        _cleanup_json_files(project_root, results)

    return results


def is_migration_complete(db_path: Optional[str] = None) -> bool:
    """Check if JSON migration has been completed.

    Returns:
        True if migration is complete
    """
    try:
        init_db(db_path)
        conn = get_connection(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM app_state WHERE key = 'json_migration_complete'")
        row = cursor.fetchone()
        return row is not None and row["value"] == "true"
    except Exception:
        return False


def _migrate_settings(project_root: str, db_path: str) -> Dict[str, Any]:
    """Migrate ff_settings.json to settings table."""
    result = {"migrated": False, "error": None}

    settings_path = os.path.join(project_root, "ff_settings.json")
    if not os.path.exists(settings_path):
        result["error"] = "ff_settings.json not found"
        return result

    try:
        with open(settings_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            result["error"] = "Invalid settings format"
            return result

        # Migrate each section
        for section, values in data.items():
            if isinstance(values, dict):
                for key, value in values.items():
                    set_setting(f"{section}.{key}", value, db_path)
            else:
                set_setting(section, values, db_path)

        result["migrated"] = True

    except Exception as e:
        result["error"] = str(e)

    return result


def _migrate_training_configs(project_root: str, db_path: str) -> Dict[str, Any]:
    """Migrate saved_configs/*.json to training_configs table."""
    result = {"migrated": 0, "errors": []}

    configs_dir = os.path.join(project_root, "src", "saved_configs")
    if not os.path.exists(configs_dir):
        # Try alternate location
        configs_dir = os.path.join(project_root, "saved_configs")

    if not os.path.exists(configs_dir):
        result["errors"].append("saved_configs directory not found")
        return result

    try:
        files = [f for f in os.listdir(configs_dir) if f.lower().endswith(".json")]
    except Exception as e:
        result["errors"].append(f"Failed to list configs: {e}")
        return result

    for filename in files:
        try:
            filepath = os.path.join(configs_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                config = json.load(f)

            if isinstance(config, dict):
                save_training_config(filename, config, db_path)
                result["migrated"] += 1
        except Exception as e:
            result["errors"].append(f"{filename}: {e}")

    # Migrate last used config marker
    last_used_path = os.path.join(configs_dir, ".last_used_config")
    if os.path.exists(last_used_path):
        try:
            with open(last_used_path, "r", encoding="utf-8") as f:
                last_used = f.read().strip()
            if last_used:
                conn = get_connection(db_path)
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO app_state (key, value)
                    VALUES ('last_used_config', ?)
                """,
                    (last_used,),
                )
                conn.commit()
        except Exception:
            pass

    return result


def _migrate_scraped_data(project_root: str, db_path: str) -> Dict[str, Any]:
    """Migrate scraped JSON files to scraped_pairs table."""
    result = {"migrated": 0, "sessions": 0, "errors": []}

    # Common scraped data file names
    scraped_files = [
        "scraped_training_data.json",
        "reddit_pairs.json",
        "fourchan_pairs.json",
        "stackexchange_pairs.json",
    ]

    for filename in scraped_files:
        filepath = os.path.join(project_root, filename)
        if not os.path.exists(filepath):
            continue

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, list):
                result["errors"].append(f"{filename}: not a list")
                continue

            # Determine source from filename
            if "reddit" in filename.lower():
                source = "reddit"
            elif "fourchan" in filename.lower() or "4chan" in filename.lower():
                source = "4chan"
            elif "stackexchange" in filename.lower():
                source = "stackexchange"
            else:
                source = "unknown"

            # Create session
            session_id = create_scrape_session(
                source=source,
                source_details=f"Migrated from {filename}",
                metadata={"migrated_from": filename},
                db_path=db_path,
            )
            result["sessions"] += 1

            # Add pairs - handle both standard and ChatML formats
            pairs = []
            for item in data:
                if isinstance(item, dict):
                    # Standard format: {"input": ..., "output": ...}
                    input_text = item.get("input", "")
                    output_text = item.get("output", "")

                    # ChatML format: {"messages": [{"role": "user", "content": ...}, ...]}
                    if not input_text and not output_text and "messages" in item:
                        messages = item.get("messages", [])
                        if isinstance(messages, list) and len(messages) >= 2:
                            # Extract first user and assistant messages
                            user_msg = None
                            assistant_msg = None
                            for msg in messages:
                                if isinstance(msg, dict):
                                    role = msg.get("role", "")
                                    content = msg.get("content", "")
                                    if role == "user" and user_msg is None:
                                        user_msg = content
                                    elif role == "assistant" and user_msg is not None and assistant_msg is None:
                                        assistant_msg = content
                                        break
                            if user_msg and assistant_msg:
                                input_text = user_msg
                                output_text = assistant_msg

                    if input_text and output_text:
                        pairs.append({"input": input_text, "output": output_text})

            if pairs:
                added = add_scraped_pairs(session_id, pairs, db_path)
                result["migrated"] += added

        except Exception as e:
            result["errors"].append(f"{filename}: {e}")

    return result


def _cleanup_json_files(project_root: str, results: Dict[str, Any]) -> None:
    """Delete JSON files after successful migration."""
    # Only delete if migration was successful
    if results["settings"].get("migrated"):
        try:
            os.remove(os.path.join(project_root, "ff_settings.json"))
        except Exception:
            pass

    # Don't delete training configs or scraped data automatically
    # as they may be needed for backup purposes


def export_all_to_json(output_dir: str, db_path: Optional[str] = None) -> Dict[str, str]:
    """Export all database data back to JSON files.

    Useful for backup or compatibility with external tools.

    Args:
        output_dir: Directory to write JSON files
        db_path: Optional database path

    Returns:
        Dictionary mapping data type to output file path
    """
    from .settings import get_all_settings
    from .training_configs import list_training_configs, get_training_config
    from .scraped_data import get_all_pairs

    os.makedirs(output_dir, exist_ok=True)
    outputs = {}

    # Export settings
    settings = get_all_settings(db_path)
    if settings:
        # Reconstruct nested structure
        nested = {}
        for key, value in settings.items():
            parts = key.split(".", 1)
            if len(parts) == 2:
                section, subkey = parts
                if section not in nested:
                    nested[section] = {}
                nested[section][subkey] = value
            else:
                nested[key] = value

        settings_path = os.path.join(output_dir, "ff_settings.json")
        with open(settings_path, "w", encoding="utf-8") as f:
            json.dump(nested, f, ensure_ascii=False, indent=2)
        outputs["settings"] = settings_path

    # Export training configs
    configs_dir = os.path.join(output_dir, "saved_configs")
    os.makedirs(configs_dir, exist_ok=True)
    for name in list_training_configs(db_path=db_path):
        config = get_training_config(name, db_path)
        if config:
            config_path = os.path.join(configs_dir, name)
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
    outputs["training_configs"] = configs_dir

    # Export scraped data
    pairs = get_all_pairs(db_path=db_path)
    if pairs:
        pairs_path = os.path.join(output_dir, "scraped_training_data.json")
        with open(pairs_path, "w", encoding="utf-8") as f:
            json.dump(pairs, f, ensure_ascii=False, indent=2)
        outputs["scraped_data"] = pairs_path

    return outputs
