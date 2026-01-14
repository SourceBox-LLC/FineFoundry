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
from .training_runs import (
    create_training_run,
    get_training_run,
    list_training_runs,
    update_training_run,
    delete_training_run,
    get_run_storage_paths,
    get_latest_run,
    get_managed_storage_root,
)
from .logs import (
    DatabaseHandler,
    get_logs,
    clear_logs,
    get_log_count,
    setup_database_logging,
)
from .evaluation_runs import (
    save_evaluation_run,
    get_evaluation_run,
    get_evaluation_runs_for_training,
    get_recent_evaluation_runs,
    get_evaluation_history_for_model,
    delete_evaluation_run,
)

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
    # Training runs
    "create_training_run",
    "get_training_run",
    "list_training_runs",
    "update_training_run",
    "delete_training_run",
    "get_run_storage_paths",
    "get_latest_run",
    "get_managed_storage_root",
    # Logging
    "DatabaseHandler",
    "get_logs",
    "clear_logs",
    "get_log_count",
    "setup_database_logging",
    # Evaluation runs
    "save_evaluation_run",
    "get_evaluation_run",
    "get_evaluation_runs_for_training",
    "get_recent_evaluation_runs",
    "get_evaluation_history_for_model",
    "delete_evaluation_run",
]
