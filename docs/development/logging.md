# FineFoundry Logging Documentation

## Overview

FineFoundry uses a **database-backed logging system** that stores all application logs in SQLite. This provides better security (no sensitive data in plain text files), queryability, and centralized log management.

## Log Storage

All logs are stored in the `app_logs` table in `finefoundry.db`:

```sql
CREATE TABLE app_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT DEFAULT (datetime('now')),
    level TEXT NOT NULL,
    logger TEXT,
    message TEXT NOT NULL,
    module TEXT,
    func_name TEXT,
    line_no INTEGER,
    exc_info TEXT
);
```

Benefits over file-based logging:

- **Security**: No sensitive data in plain text log files
- **Queryability**: Filter logs by level, logger, time range
- **Centralized**: All logs in one place with the rest of app data
- **Automatic cleanup**: Easy to purge old logs

## Log Levels

The logging system uses standard Python log levels:

- **DEBUG**: Detailed diagnostic information (hidden by default)
- **INFO**: General informational messages about operations
- **WARNING**: Warning messages for potentially problematic situations
- **ERROR**: Error messages with full stack traces
- **CRITICAL**: Critical errors that may cause application failure

## Enabling Debug Mode

To see DEBUG level logs (more detailed information), set the environment variable:

```bash
export FINEFOUNDRY_DEBUG=1
uv run src/main.py
```

Or on Windows:

```cmd
set FINEFOUNDRY_DEBUG=1
uv run src/main.py
```

## Log Format

Each log entry includes:

```
2025-01-15 14:30:45 - module.name - LEVEL - [filename.py:123] - Message
```

Example:

```
2025-01-15 14:30:45 - __main__ - INFO - [main.py:2059] - Starting scrape operation
2025-01-15 14:30:45 - helpers.merge - INFO - [merge.py:170] - Starting merge operation
2025-01-15 14:30:50 - helpers.merge - INFO - [merge.py:341] - Merged 1250 records into session: combined_dataset
```

## What Gets Logged

### Merge Operations

- Start of merge operation
- Validation results
- Dataset loading progress
- File/directory operations
- Success/failure status
- Number of records merged

### Download Operations

- Download button clicks
- Source and destination paths
- File copy operations
- Success/failure status
- Errors with full stack traces

### Error Handling

- All errors include full stack traces (via `exc_info=True`)
- Source and destination paths for debugging
- Operation context (what was being attempted)

## Viewing Logs

### Programmatic Access

```python
from db import get_logs, get_log_count, clear_logs

# Get recent logs
logs = get_logs(limit=100)

# Filter by level
errors = get_logs(level="ERROR", limit=50)

# Filter by logger
merge_logs = get_logs(logger="helpers.merge", limit=100)

# Get log count
error_count = get_log_count(level="ERROR")

# Clear old logs
clear_logs(older_than_days=30)
```

### Console Output

Logs are also printed to the console in real-time with this format:

```
2025-01-15 14:30:45 - module.name - LEVEL - [filename.py:123] - Message
```

### Querying with SQL

You can also query logs directly with SQLite:

```bash
sqlite3 finefoundry.db "SELECT * FROM app_logs WHERE level='ERROR' ORDER BY timestamp DESC LIMIT 10"
```

## Adding Logging to New Modules

To add logging to a new module:

```python
from helpers.logging_config import get_logger

# At module level
logger = get_logger(__name__)

# In your functions
def my_function():
    logger.info("Starting operation")
    try:
        # Your code here
        logger.debug("Detailed debug info")
    except Exception as e:
        logger.error("Operation failed", exc_info=True)
```

## Log Management

### Clearing Old Logs

```python
from db import clear_logs

# Clear logs older than 30 days
clear_logs(older_than_days=30)

# Clear all logs
clear_logs(older_than_days=0)
```

### Database Size

Logs are stored efficiently in SQLite. For most use cases, the database size remains manageable. If needed, clear old logs periodically.

## Troubleshooting

### Logs not appearing

1. Check that `finefoundry.db` exists in the project root
1. Verify the database is initialized: `from db import init_db; init_db()`
1. Check if DEBUG mode is needed: `export FINEFOUNDRY_DEBUG=1`

### Too many logs

1. Use `clear_logs(older_than_days=N)` to purge old entries
1. Default log level is INFO, which is reasonable for most use cases

### Finding specific issues

1. Query ERROR level logs: `get_logs(level="ERROR")`
1. Filter by logger name for specific modules
1. Stack traces are included in the `exc_info` field

## Best Practices

1. **Use appropriate log levels**:

   - DEBUG: Verbose diagnostic info
   - INFO: Normal operation milestones
   - WARNING: Potential issues
   - ERROR: Actual failures

1. **Include context**: Log relevant variables and paths

1. **Use `exc_info=True`** for exceptions to capture full stack traces

1. **Log operation boundaries**: Start and completion of major operations

1. **Don't log sensitive data**: Avoid logging passwords, API keys, etc.
