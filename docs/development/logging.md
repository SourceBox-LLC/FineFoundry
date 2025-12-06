# FineFoundry Logging Documentation

## Overview

FineFoundry now has a comprehensive, professional logging system that tracks all important operations, errors, and debug information. Logs are automatically rotated and stored in the `logs/` directory.

## Log Files

All log files are stored in the `logs/` directory in the project root:

- `__main__.log` - Main application logs
- `helpers_merge.log` - Dataset merge operation logs
- Additional module-specific logs as needed

Each log file:

- Automatically rotates when it reaches 10MB
- Keeps 5 backup files
- Uses UTF-8 encoding
- Includes timestamps, log levels, source file/line numbers

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
2025-01-15 14:30:45 - __main__ - INFO - [main.py:2059] - Download merged dataset called with destination: /home/user/Downloads
2025-01-15 14:30:45 - helpers.merge - INFO - [merge.py:170] - Starting merge operation
2025-01-15 14:30:50 - helpers.merge - INFO - [merge.py:341] - Saving 1250 merged records to JSON: /home/user/Projects/merged_dataset.json
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

### Real-time Monitoring

```bash
# Watch all main application logs
tail -f logs/__main__.log

# Watch merge operation logs
tail -f logs/helpers_merge.log

# Watch all logs
tail -f logs/*.log
```

### Searching Logs

```bash
# Find all errors
grep "ERROR" logs/*.log

# Find specific operation
grep "Download merged" logs/__main__.log

# Find logs from a specific date
grep "2025-01-15" logs/*.log
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

## Log Rotation

Logs automatically rotate when they reach 10MB. The system keeps:

- Current log file: `module_name.log`
- 5 backups: `module_name.log.1` through `module_name.log.5`

Older backups are automatically deleted.

## Troubleshooting

### Logs not appearing

1. Check that the `logs/` directory exists (it's auto-created)
1. Verify file permissions
1. Check if DEBUG mode is needed: `export FINEFOUNDRY_DEBUG=1`

### Too many logs

1. Reduce log level (default is INFO, which is reasonable)
1. The rotation system automatically manages disk space
1. Manually delete old `.log.N` backup files if needed

### Finding specific issues

1. Check ERROR level logs first: `grep "ERROR" logs/*.log`
1. Look for the specific operation (merge, download, etc.)
1. Stack traces are included for all errors

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
