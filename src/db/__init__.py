"""FineFoundry SQLite database module.

Provides unified storage for settings, training configs, scraped data,
and scrape sessions. Replaces scattered JSON files with a single database.
"""

from .core import (
    get_db_path,
    get_connection,
    init_db,
    close_all_connections,
)
from .settings import (
    get_setting,
    set_setting,
    get_all_settings,
    delete_setting,
)
from .training_configs import (
    list_training_configs,
    get_training_config,
    save_training_config,
    delete_training_config,
    rename_training_config,
    get_last_used_config,
    set_last_used_config,
    validate_config,
)
from .scraped_data import (
    create_scrape_session,
    add_scraped_pairs,
    get_scrape_session,
    list_scrape_sessions,
    get_pairs_for_session,
    export_session_to_json,
    delete_scrape_session,
)
from .migrate import migrate_from_json

__all__ = [
    # Core
    "get_db_path",
    "get_connection",
    "init_db",
    "close_all_connections",
    # Settings
    "get_setting",
    "set_setting",
    "get_all_settings",
    "delete_setting",
    # Training configs
    "list_training_configs",
    "get_training_config",
    "save_training_config",
    "delete_training_config",
    "rename_training_config",
    "get_last_used_config",
    "set_last_used_config",
    "validate_config",
    # Scraped data
    "create_scrape_session",
    "add_scraped_pairs",
    "get_scrape_session",
    "list_scrape_sessions",
    "get_pairs_for_session",
    "export_session_to_json",
    "delete_scrape_session",
    # Migration
    "migrate_from_json",
]
