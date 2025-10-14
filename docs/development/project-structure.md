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
│   ├── helpers/              # Helper modules
│   │   ├── common.py         # Common utilities
│   │   ├── logging_config.py # Logging configuration
│   │   ├── merge.py          # Dataset merging logic
│   │   ├── scrape.py         # Scraping helpers
│   │   ├── training.py       # Training helpers
│   │   ├── training_pod.py   # Runpod training helpers
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
│   │   └── tabs/             # Tab-specific UI
│   │       ├── tab_scrape.py
│   │       ├── tab_build.py
│   │       ├── tab_training.py
│   │       ├── tab_merge.py
│   │       ├── tab_analysis.py
│   │       ├── tab_settings.py
│   │       ├── scrape/       # Scrape tab sections
│   │       ├── merge/        # Merge tab sections
│   │       └── training/     # Training tab sections
│   └── runpod/               # Runpod integration
│       ├── runpod_pod.py     # Pod management
│       └── ensure_infra.py   # Infrastructure setup
├── logs/                      # Log files (auto-created)
├── img/                       # Images and assets
├── requirements.txt           # Python dependencies
├── uv.lock                    # UV lock file
└── README.md                  # Main README

```

## Core Modules

### Main Application (`src/main.py`)

The main entry point that:
- Initializes the Flet desktop application
- Creates all UI tabs
- Sets up event handlers
- Manages application state
- Coordinates between UI and business logic

**Key sections:**
- Imports and logger setup (lines 1-50)
- Helper functions and utilities (lines 100-500)
- Tab-specific UI creation (lines 500-3000)
- Main app function (bottom of file)

### Helpers (`src/helpers/`)

Business logic separated from UI:

#### `common.py`
- `safe_update()` - Safe UI update wrapper
- `set_terminal_title()` - Terminal title management
- Utility functions used across modules

#### `logging_config.py`
- Centralized logging setup
- Rotating file handlers
- Console and file logging
- Debug mode support
- See [Logging Guide](logging.md)

#### `merge.py`
- `run_merge()` - Main merge operation
- `preview_merged()` - Dataset preview
- Dataset loading and column mapping
- Interleave and concatenate operations
- JSON and HF dataset handling

#### `scrape.py`
- `run_reddit_scrape()` - Reddit scraping wrapper
- `run_real_scrape()` - 4chan scraping wrapper
- `run_stackexchange_scrape()` - Stack Exchange wrapper
- UI update helpers for scraping progress

#### `training.py`
- `run_local_training()` - Local Docker training
- `stop_local_training()` - Stop training
- `build_hp_from_controls()` - Hyperparameter extraction
- Training configuration management

#### `training_pod.py`
- `run_pod_training()` - Runpod training orchestration
- `restart_pod_container()` - Pod restart
- `open_runpod()` / `open_web_terminal()` - Web interfaces
- `copy_ssh_command()` - SSH access helper
- `ensure_infrastructure()` - Volume and template setup
- Teardown operations

#### `build.py`
- `run_build()` - Dataset building from JSON
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
- `tab_scrape.py` - Composes scrape tab sections
- `tab_build.py` - Composes build/publish sections
- `tab_training.py` - Composes training sections
- `tab_merge.py` - Composes merge sections
- `tab_analysis.py` - Composes analysis sections
- `tab_settings.py` - Composes settings sections

Each tab builder:
1. Imports section builders
2. Receives controls from main.py
3. Calls section builders
4. Returns composed layout

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
User (UI) → main.py event handler
           ↓
         helpers/scrape.py wrapper
           ↓
         scrapers/{source}_scraper.py
           ↓
         JSON file written
           ↓
         UI updated with progress
```

### Building Flow
```
JSON file → helpers/build.py
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
Multiple sources → helpers/merge.py
                  ↓
                Column mapping
                  ↓
                Dataset loading
                  ↓
                Concatenate/Interleave
                  ↓
                Save (JSON or HF format)
                  ↓
                Preview generation
```

### Training Flow
```
Configuration (UI) → helpers/training.py or helpers/training_pod.py
                    ↓
                  Runpod pod creation OR Local Docker
                    ↓
                  train.py execution (in container)
                    ↓
                  Checkpoints to /data or local volume
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
- **Settings**: Persisted in `.env` or OS-specific config

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
2. Create sections in `src/ui/tabs/newtab/sections/`
3. Add business logic to `src/helpers/` if needed
4. Import and integrate in `src/main.py`
5. Add documentation in `docs/user-guide/`

### Adding a New Scraper

1. Create `src/scrapers/newsource_scraper.py`
2. Implement `scrape()` function
3. Add wrapper in `src/helpers/scrape.py`
4. Add UI in scrape tab sections
5. Document in API reference

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
tests/                    # Test directory (create if needed)
├── unit/                # Unit tests
├── integration/         # Integration tests
└── fixtures/            # Test data
```

See [Testing Guide](testing.md) for details.

## Configuration Files

- `requirements.txt` - Python dependencies
- `uv.lock` - UV package lock file
- `.env` (not in repo) - Local environment variables
- Settings stored in OS-specific locations via Flet

## Build Artifacts

- `logs/` - Log files (git ignored)
- `*.json` - Scraped data (git ignored)
- `hf_dataset/` - Built datasets (git ignored)
- `__pycache__/` - Python cache (git ignored)
- `.venv/` - Virtual environment (git ignored)

## Related Documentation

- [Contributing Guide](contributing.md)
- [Logging Guide](logging.md)
- [Testing Guide](testing.md)
- [API Reference](../api/scrapers.md)

---

**Back to**: [Development Documentation](../README.md#-development) | [Documentation Index](../README.md)
