# Database Architecture

FineFoundry uses SQLite for unified data storage, replacing the previous scattered JSON files. This provides better data integrity, queryability, and a single source of truth for all application data.

## Overview

The database file `finefoundry.db` is created in the project root on first run. It stores:

- **Settings** — Application configuration (HF token, RunPod API key, Ollama, proxy)
- **Training Configs** — Saved hyperparameter configurations
- **Scrape Sessions** — Metadata for each scrape run
- **Scraped Pairs** — Individual input/output pairs from scraping
- **App State** — Internal state (last used config, schema version)

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
    source TEXT NOT NULL,           -- "4chan", "reddit", "stackexchange"
    source_details TEXT,            -- e.g., "boards=pol,b" or "url=..."
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

### app_state

Internal application state.

```sql
CREATE TABLE app_state (
    key TEXT PRIMARY KEY,
    value TEXT
);
```

Used for: `last_used_config`, `schema_version`, `json_migration_complete`.

## Module Structure

```
src/db/
├── __init__.py          # Public exports
├── core.py              # Connection management, schema init
├── settings.py          # Settings CRUD
├── training_configs.py  # Training config CRUD
├── scraped_data.py      # Scrape sessions and pairs CRUD
└── migrate.py           # JSON to SQLite migration
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

### Migration

Existing JSON files are automatically migrated on first run:

```python
from db import init_db, migrate_from_json
from db.migrate import is_migration_complete

init_db()
if not is_migration_complete():
    results = migrate_from_json()
    print(results)
```

The migration handles:
- `ff_settings.json` → `settings` table
- `saved_configs/*.json` → `training_configs` table
- `scraped_training_data.json` → `scraped_pairs` table (both standard and ChatML formats)

### Export to JSON

For compatibility with external tools:

```python
from db.migrate import export_all_to_json

outputs = export_all_to_json("/path/to/backup")
# Returns: {"settings": "...", "training_configs": "...", "scraped_data": "..."}
```

## Thread Safety

The database uses thread-local connections, making it safe to use from multiple threads. Each thread gets its own connection that is reused for the lifetime of the thread.

## Backward Compatibility

- JSON files are still written during scraping for compatibility with the Build & Publish workflow
- The `helpers/training_config.py` module falls back to filesystem if database operations fail
- Existing JSON files are preserved after migration (not deleted)

## File Locations

| File | Purpose |
|------|---------|
| `finefoundry.db` | Main SQLite database (project root) |
| `ff_settings.json` | Legacy settings (migrated, kept for backup) |
| `saved_configs/*.json` | Legacy training configs (migrated) |
| `scraped_training_data.json` | Scrape output (still written for compatibility) |
