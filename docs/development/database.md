# Database Architecture

FineFoundry uses SQLite as the **sole storage mechanism** for all application data. There are no filesystem fallbacks or legacy JSON files.

## Overview

The database file `finefoundry.db` is created in the project root on first run. It stores:

- **Settings** — Application configuration (HF token, RunPod API key, Ollama, proxy)
- **Training Configs** — Saved hyperparameter configurations
- **Scrape Sessions** — Metadata for each scrape run
- **Scraped Pairs** — Individual input/output pairs from scraping
- **Training Runs** — Managed training runs with logs, adapters, checkpoints
- **App Logs** — Application logs (replaces file-based logging)
- **App State** — Internal state (last used config, schema version)

The only file-based storage is `training_outputs/` for binary model artifacts (checkpoints, adapters) which cannot be stored in SQLite.

## Schema

### settings

Key-value store for application settings.

```sql
CREATE TABLE settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT DEFAULT (datetime('now'))
);
```

Keys use dot notation for namespacing: `huggingface.token`, `runpod.api_key`, `ollama.enabled`, etc.

### training_configs

Saved training configurations.

```sql
CREATE TABLE training_configs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    config_json TEXT NOT NULL,
    train_target TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);
```

### scrape_sessions

Metadata for each scrape run.

```sql
CREATE TABLE scrape_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,           -- "4chan", "reddit", "stackexchange", "synthetic"
    source_details TEXT,            -- e.g., "boards=pol,b" or "url=..." or "sources=3, type=qa, model=..."
    dataset_format TEXT DEFAULT 'standard',
    pair_count INTEGER DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now')),
    metadata_json TEXT
);
```

### scraped_pairs

Individual input/output pairs linked to sessions.

```sql
CREATE TABLE scraped_pairs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    input_text TEXT NOT NULL,
    output_text TEXT NOT NULL,
    source_url TEXT,
    metadata_json TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (session_id) REFERENCES scrape_sessions(id) ON DELETE CASCADE
);
```

### training_runs

Managed training runs with metadata and file paths.

```sql
CREATE TABLE training_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    status TEXT DEFAULT 'pending',    -- pending, running, completed, failed, cancelled
    base_model TEXT,
    dataset_source TEXT,              -- "huggingface", "database"
    dataset_id TEXT,                  -- HF repo or session ID
    storage_path TEXT,                -- Path to training_outputs/<run_name>/
    output_dir TEXT,
    adapter_path TEXT,
    checkpoint_path TEXT,
    hp_json TEXT,                     -- Hyperparameters JSON
    logs_json TEXT,                   -- Training logs JSON
    started_at TEXT,
    completed_at TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now')),
    metadata_json TEXT
);
```

### app_logs

Application logs stored in the database (replaces file-based logging).

```sql
CREATE TABLE app_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT DEFAULT (datetime('now')),
    level TEXT NOT NULL,              -- DEBUG, INFO, WARNING, ERROR, CRITICAL
    logger TEXT,                      -- Logger name (e.g., "finefoundry.helpers.scrape")
    message TEXT NOT NULL,
    module TEXT,
    func_name TEXT,
    line_no INTEGER,
    exc_info TEXT                     -- Exception traceback if any
);
```

### app_state

Internal application state.

```sql
CREATE TABLE app_state (
    key TEXT PRIMARY KEY,
    value TEXT
);
```

Used for: `last_used_config`, `schema_version`.

## Module Structure

```
src/db/
├── __init__.py          # Public exports
├── core.py              # Connection management, schema init
├── settings.py          # Settings CRUD
├── training_configs.py  # Training config CRUD
├── scraped_data.py      # Scrape sessions and pairs CRUD
├── training_runs.py     # Training runs CRUD
└── logs.py              # Database logging handler
```

## Usage

### Basic Operations

```python
from db import init_db, get_setting, set_setting

# Initialize database (safe to call multiple times)
init_db()

# Settings
set_setting("huggingface.token", "hf_xxx")
token = get_setting("huggingface.token")

# Training configs
from db import save_training_config, get_training_config, list_training_configs

save_training_config("my_config.json", {"hp": {...}, "meta": {...}})
config = get_training_config("my_config.json")
all_configs = list_training_configs()

# Scraped data
from db import create_scrape_session, add_scraped_pairs, get_pairs_for_session

session_id = create_scrape_session(
    source="reddit",
    source_details="r/LocalLLaMA",
    dataset_format="standard"
)
add_scraped_pairs(session_id, [
    {"input": "Hello", "output": "Hi there!"},
    {"input": "How are you?", "output": "I'm fine!"},
])
pairs = get_pairs_for_session(session_id)
```

### Training Runs

```python
from db.training_runs import (
    create_training_run,
    get_training_run,
    update_training_run,
    delete_training_run,
    list_training_runs,
    get_managed_storage_root,
)

# Create a new training run (auto-creates storage directory)
run = create_training_run(
    name="my_finetune",
    base_model="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    dataset_source="database",
    dataset_id="123",
)
# run["storage_path"] = "training_outputs/my_finetune_20241209_123456/"

# Update status
update_training_run(run["id"], status="running")

# List all runs
runs = list_training_runs(status="completed")
```

### Logging

```python
from db import get_logs, get_log_count, clear_logs

# Get recent logs
logs = get_logs(limit=100, level="ERROR")

# Get log count
count = get_log_count(level="WARNING")

# Clear old logs
clear_logs(older_than_days=30)
```

### Export to JSON

For compatibility with external tools:

```python
from helpers.scrape_db import export_to_json

# Export a scrape session to JSON
export_to_json(session_id, "/path/to/output.json")
```

## Thread Safety

The database uses thread-local connections, making it safe to use from multiple threads. Each thread gets its own connection that is reused for the lifetime of the thread.

## File Locations

| Path | Purpose |
|------|---------|
| `finefoundry.db` | Main SQLite database (project root) |
| `training_outputs/` | Binary model artifacts (checkpoints, adapters) |

### Temporary Files

- **Synthetic data generation** uses OS temp directories (`/tmp/finefoundry_synth_*`) that are automatically cleaned up after generation completes.
- **Docker/RunPod training** exports database sessions to temporary JSON files in mounted directories (required by Unsloth trainer).

### Optional Exports

Users can optionally export data to JSON for external tools:

- Merged datasets can be exported to JSON via the Merge tab
- Scrape sessions can be exported via `helpers.scrape_db.export_to_json()`
- Training logs can be saved to files via the Training tab
