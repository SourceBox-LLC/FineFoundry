# Project Structure

Understanding the FineFoundry codebase organization.

## Directory Overview

```
FineFoundry-Core/
├── docs/                       # Documentation
│   ├── user-guide/            # User-facing documentation
│   ├── development/           # Developer documentation
│   ├── api/                   # API reference
│   └── deployment/            # Deployment guides
├── src/                       # Source code
│   ├── main.py               # Main application entry point
│   ├── save_dataset.py       # Dataset builder CLI
│   ├── synthetic_cli.py      # Synthetic data generation CLI
│   ├── db/                   # Database module (sole storage)
│   │   ├── __init__.py       # Public exports
│   │   ├── core.py           # Connection management, schema
│   │   ├── settings.py       # Settings CRUD
│   │   ├── training_configs.py # Training config CRUD
│   │   ├── scraped_data.py   # Scrape sessions and pairs
│   │   ├── training_runs.py  # Training runs CRUD
│   │   └── logs.py           # Database logging handler
│   ├── helpers/              # Helper modules
│   │   ├── common.py         # Common utilities
│   │   ├── logging_config.py # Database-backed logging
│   │   ├── settings.py       # Settings helper (database)
│   │   ├── settings_ollama.py# Ollama settings (database)
│   │   ├── training_config.py# Training config helper (database)
│   │   ├── scrape_db.py      # Scrape data helper (database)
│   │   ├── merge.py          # Dataset merging logic
│   │   ├── scrape.py         # Scraping helpers
│   │   ├── synthetic.py      # Synthetic data generation
│   │   ├── training.py       # Training helpers
│   │   ├── training_pod.py   # Runpod training helpers
│   │   ├── local_inference.py# Local inference helpers
│   │   ├── build.py          # Dataset building helpers
│   │   ├── boards.py         # Board listing
│   │   ├── datasets.py       # Dataset utilities
│   │   ├── theme.py          # UI theme and styling
│   │   ├── ui.py             # UI component helpers
│   │   └── proxy.py          # Proxy configuration
│   ├── scrapers/             # Data collection modules
│   │   ├── fourchan_scraper.py
│   │   ├── reddit_scraper.py
│   │   └── stackexchange_scraper.py
│   ├── ui/                   # UI components
│   │   └── tabs/             # Tab-specific UI (layouts + controllers)
│   │       ├── tab_*.py      # Tab layout builders
│   │       ├── *_controller.py # Tab controllers
│   │       └── */sections/   # Tab section modules
│   └── runpod/               # Runpod integration
│       ├── runpod_pod.py     # Pod management
│       └── ensure_infra.py   # Infrastructure setup
├── tests/                     # Test suite
│   ├── unit/                 # Unit tests
│   └── integration/          # Integration tests
├── training_outputs/          # Model artifacts (checkpoints, adapters)
├── finefoundry.db            # SQLite database (auto-created)
├── img/                       # Images and assets
├── pyproject.toml            # Project configuration
├── uv.lock                    # UV lock file
└── README.md                  # Main README

```

## Core Modules

### Main Application (`src/main.py`)

The main entry point that:

- Initializes the Flet desktop application
- Sets up the global app shell (AppBar, welcome view, shared settings/proxy/HF/Runpod/Ollama controls)
- Delegates tab wiring to dedicated controllers in `ui/tabs/*_controller.py`
- Coordinates between controllers and shared helper modules

**Key responsibilities (after controller refactor):**

- Imports and logger setup
- Global helpers (user guide dialog, keyboard shortcuts, settings I/O)
- Building each tab by calling `build_*_tab_with_logic(...)` from the appropriate controller
- Adding all tabs to the Flet `Tabs` control and bootstrapping the app

### Helpers (`src/helpers/`)

Business logic separated from UI:

#### `common.py`

- `safe_update()` - Safe UI update wrapper
- `set_terminal_title()` - Terminal title management
- Utility functions used across modules

#### `logging_config.py`

- Database-backed logging via `DatabaseHandler`
- Console output for real-time monitoring
- Debug mode support via `FINEFOUNDRY_DEBUG` env var
- See [Logging Guide](logging.md)

#### `merge.py`

- `run_merge()` - Main merge operation
- `preview_merged()` - Dataset preview
- Dataset loading and column mapping
- Interleave and concatenate operations
- Database session and HF dataset handling
- Merged results saved to database with optional JSON export

#### `scrape.py`

- `run_reddit_scrape()` - Reddit scraping wrapper
- `run_real_scrape()` - 4chan scraping wrapper
- `run_stackexchange_scrape()` - Stack Exchange wrapper
- UI update helpers for scraping progress

#### `synthetic.py`

- `run_synthetic_generation()` - Synthetic data generation using Unsloth's SyntheticDataKit
- Document ingestion (PDF, DOCX, TXT, HTML, URLs)
- Q&A, chain-of-thought, and summary generation
- Local LLM serving via vLLM
- Database integration for generated pairs

#### `training.py`

- `run_local_training()` - Native local training via Python subprocess
- `stop_local_training()` - Stop local training subprocess
- `build_hp_from_controls()` - Hyperparameter extraction shared between Runpod and local

#### `training_config.py`

- `list_saved_configs()` - List configs from database
- `save_config()` / `read_json_file()` / `delete_config()` - Config CRUD operations
- `rename_config()` - Rename existing configs
- `get_last_used_config_name()` / `set_last_used_config_name()` - Track last-used config for auto-load on startup
- All configs stored in SQLite database (no filesystem fallback)

#### `local_inference.py`

- Helpers for loading a base model + LoRA adapter locally (used by Quick Local Inference)
- Caches loaded models to speed up repeated generations
- Inference tab helpers for global inference over fine-tuned adapters

#### `training_pod.py`

- `run_pod_training()` - Runpod training orchestration
- `restart_pod_container()` - Pod restart
- `open_runpod()` / `open_web_terminal()` - Web interfaces
- `copy_ssh_command()` - SSH access helper
- `ensure_infrastructure()` - Volume and template setup
- Teardown operations

#### `build.py`

- `run_build()` - Dataset building from database sessions or HF datasets
- `run_push_async()` - Async Hub push
- Split creation and validation
- Dataset card generation

#### `datasets.py`

- `guess_input_output_columns()` - Column detection
- Dataset schema utilities

#### `theme.py`

- Color definitions
- Icon mappings
- Styling constants
- Accent colors and borders

#### `ui.py`

- `pill()` - Chip/pill components
- `section_title()` - Section headers
- `make_wrap()` - Wrap containers
- `make_selectable_pill()` - Selectable chips
- `two_col_row()` / `two_col_header()` - Two-column layouts
- `compute_two_col_flex()` - Column width calculations

#### `proxy.py`

- `apply_proxy_from_env()` - Environment-based proxy setup
- Proxy configuration helpers for all scrapers

### Scrapers (`src/scrapers/`)

Data collection modules:

#### `fourchan_scraper.py`

- `scrape()` - Main scraping function
- `fetch_catalog_pages()` - Catalog fetching
- `fetch_thread()` - Thread fetching
- `build_pairs_normal()` - Adjacent pairing
- `build_pairs_contextual()` - Context-aware pairing
- Text cleaning and normalization
- Quote chain and cumulative strategies

#### `reddit_scraper.py`

- CLI entry point
- `crawl()` - Subreddit/post crawling
- `expand_more_comments()` - Comment expansion
- `build_pairs_parent_child()` - Parent-child pairing
- `build_pairs_contextual()` - Context-based pairing
- Comment tree traversal

#### `stackexchange_scraper.py`

- `scrape()` - Q&A pair scraping
- Stack Exchange API integration
- Backoff handling
- HTML cleaning

### UI Components (`src/ui/`)

Organized by tab and section for modularity:

#### Tab Builders (`src/ui/tabs/`)

- `tab_scrape.py` - Composes Data Sources tab sections
- `tab_build.py` - Composes publish sections
- `tab_training.py` - Composes training sections
- `tab_inference.py` - Composes the Inference tab (global inference over fine-tuned adapters)
- `tab_merge.py` - Composes merge sections
- `tab_analysis.py` - Composes analysis sections
- `tab_settings.py` - Composes settings sections

Each tab builder:

1. Imports section builders
1. Receives controls from a tab controller (for example, `ui/tabs/scrape_controller.py`, `ui/tabs/build_controller.py`, `ui/tabs/merge_controller.py`, `ui/tabs/analysis_controller.py`, `ui/tabs/training_controller.py`, or `ui/tabs/inference_controller.py`)
1. Calls section builders
1. Returns the composed layout

Tab controllers own behavior and state for each tab and then delegate layout to the corresponding builder:

- `ui/tabs/scrape_controller.py` → `tab_scrape.py`
- `ui/tabs/build_controller.py` → `tab_build.py`
- `ui/tabs/merge_controller.py` → `tab_merge.py`
- `ui/tabs/analysis_controller.py` → `tab_analysis.py`
- `ui/tabs/training_controller.py` → `tab_training.py`
- `ui/tabs/inference_controller.py` → `tab_inference.py`

`src/main.py` now calls the exported `build_*_tab_with_logic(...)` functions from these controllers instead of creating tab controls and handlers inline.

#### Section Builders

Example: `src/ui/tabs/scrape/sections/`

- `source_section.py` - Data source selector
- `boards_section.py` - Board selection
- `params_section.py` - Parameters
- `progress_section.py` - Progress indicators
- `log_section.py` - Log output
- `preview_section.py` - Dataset preview

Benefits:

- Clear separation of concerns
- Easy to test individual sections
- Maintainable codebase
- Reusable components

### Runpod Integration (`src/runpod/`)

#### `runpod_pod.py`

- `create_pod()` - Pod creation
- `get_pod()` - Pod status
- `stop_pod()` - Pod termination
- `list_pods()` - List user pods
- `pod_logs()` - Stream logs
- API wrapper functions

#### `ensure_infra.py`

- `ensure_network_volume()` - Volume creation/reuse
- `ensure_pod_template()` - Template creation/update
- Infrastructure validation
- API key management

## Data Flow

### Scraping Flow

```
User (UI) → Data Sources tab controller (`ui/tabs/scrape_controller.py`)
           ↓
         helpers/scrape.py wrapper
           ↓
         scrapers/{source}_scraper.py
           ↓
         Database session created (db/scraped_data.py)
           ↓
         UI updated with progress
```

### Building Flow

```
Database session or HF dataset → helpers/build.py
                               ↓
                             Dataset validation
                               ↓
                             Train/val/test splits
                               ↓
                             HF DatasetDict
                               ↓
                             Save to disk / Push to Hub
```

### Merging Flow

```
Multiple sources (DB sessions / HF) → helpers/merge.py
                                     ↓
                                   Column mapping
                                     ↓
                                   Dataset loading
                                     ↓
                                   Concatenate/Interleave
                                     ↓
                                   Save to database (+ optional JSON export)
                                     ↓
                                   Preview generation
```

### Training Flow

```
Configuration (UI) → helpers/training.py or helpers/training_pod.py
                    ↓
                  Runpod pod creation OR Native local subprocess
                    ↓
                  Unsloth trainer execution
                    ↓
                  Checkpoints to /data (Runpod) or local output dir
                    ↓
                  Optional Hub push
```

## State Management

FineFoundry uses simple state management:

- **UI State**: Managed by Flet controls and their values
- **Operation State**: Dictionaries passed to async functions
  - `cancel_state` - Cancellation flags
  - `merge_cancel` - Merge cancellation
- **Configuration**: Stored in UI controls, read when needed
- **Settings**: Persisted in the SQLite database (`finefoundry.db`)

No complex state management framework needed due to:

- Single-user desktop app
- Synchronous UI updates
- Clear operation boundaries

## File Naming Conventions

- **Modules**: `snake_case.py`
- **Classes**: `PascalCase`
- **Functions**: `snake_case()`
- **Constants**: `UPPER_SNAKE_CASE`
- **UI Builders**: `build_{name}_{section}()`

## Import Conventions

```python
# Standard library
import asyncio
import json
from typing import List

# Third-party
import flet as ft

# Local
from helpers.logging_config import get_logger
from helpers.common import safe_update
```

Order: standard library → third-party → local

## Adding New Features

### Adding a New Tab

1. Create `src/ui/tabs/tab_newtab.py`
1. Create sections in `src/ui/tabs/newtab/sections/`
1. Add business logic to `src/helpers/` if needed
1. Import and integrate in `src/main.py`
1. Add documentation in `docs/user-guide/`

### Adding a New Scraper

1. Create `src/scrapers/newsource_scraper.py`
1. Implement `scrape()` function
1. Add wrapper in `src/helpers/scrape.py`
1. Add UI in Data Sources tab sections
1. Document in API reference

### Adding Logging

```python
from helpers.logging_config import get_logger

logger = get_logger(__name__)

# In your functions
logger.info("Operation started")
logger.debug("Detailed info")
logger.error("Error occurred", exc_info=True)
```

See [Logging Guide](logging.md) for details.

## Testing Locations

```
tests/                    # Test directory (collected by pytest)
├── unit/                 # Unit tests for helpers, save_dataset, etc.
├── integration/          # Integration tests (end-to-end flows and UI/controller smoke tests)
└── fixtures/             # Optional shared test data
```

See [Testing Guide](testing.md) for details on test types, commands, and coverage.

## Configuration Files

- `pyproject.toml` - Project configuration and dependencies
- `uv.lock` - UV package lock file
- `.env` (not in repo) - Optional local environment variables (for example, `HF_TOKEN`)
- Settings persisted in the SQLite database (`finefoundry.db`)

## Build Artifacts

- `finefoundry.db` - SQLite database (git ignored)
- `training_outputs/` - Model artifacts (git ignored)
- `hf_dataset/` - Built datasets (git ignored)
- `__pycache__/` - Python cache (git ignored)
- `.venv/` - Virtual environment (git ignored)

## Related Documentation

- [Contributing Guide](contributing.md)
- [Logging Guide](logging.md)
- [Testing Guide](testing.md)
- [API Reference](../api/scrapers.md)

______________________________________________________________________

**Back to**: [Development Documentation](../README.md#-development) | [Documentation Index](../README.md)
