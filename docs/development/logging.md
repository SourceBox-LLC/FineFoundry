# Logging Guide

FineFoundry stores all logs in the SQLite database rather than text files. This keeps sensitive data secure, makes logs queryable, and centralizes everything with the rest of your app data.

## Where Logs Go

Logs live in the `app_logs` table in `finefoundry.db`. Each entry includes timestamp, level, logger name, message, source location, and exception info if applicable.

## Log Levels

Standard Python levels apply:

- **DEBUG** — Detailed diagnostic info (hidden by default)
- **INFO** — Normal operation milestones
- **WARNING** — Potential issues worth noting
- **ERROR** — Failures with full stack traces
- **CRITICAL** — Severe errors that may crash the app

## Enabling Debug Mode

To see DEBUG logs, set the environment variable before launching:

```bash
export FINEFOUNDRY_DEBUG=1
uv run src/main.py
```

On Windows, use `set FINEFOUNDRY_DEBUG=1`.

## Viewing Logs

Logs print to the console in real time and are stored in the database. You can query them programmatically:

```python
from db import get_logs, get_log_count, clear_logs

logs = get_logs(limit=100)                          # recent logs
errors = get_logs(level="ERROR", limit=50)          # filter by level
merge_logs = get_logs(logger="helpers.merge")       # filter by module
clear_logs(older_than_days=30)                      # cleanup
```

Or query directly with SQLite:

```bash
sqlite3 finefoundry.db "SELECT * FROM app_logs WHERE level='ERROR' ORDER BY timestamp DESC LIMIT 10"
```

## Adding Logging to Your Code

```python
from helpers.logging_config import get_logger

logger = get_logger(__name__)

def my_function():
    logger.info("Starting operation")
    try:
        # your code
        logger.debug("Detailed info for debugging")
    except Exception as e:
        logger.error("Operation failed", exc_info=True)
```

## Maintenance

Clear old logs periodically with `clear_logs(older_than_days=30)`. The database stays manageable for most use cases, but if it grows too large, purge older entries.

## Troubleshooting

If logs aren't appearing, verify `finefoundry.db` exists and the database is initialized. You may need DEBUG mode enabled to see verbose output.

If you have too many logs, use `clear_logs()` to purge old entries. The default INFO level is reasonable for most use cases.

## Best Practices

Use appropriate log levels—DEBUG for verbose diagnostics, INFO for milestones, WARNING for potential issues, ERROR for failures. Include relevant context (variables, paths) in log messages. Use `exc_info=True` for exceptions to capture stack traces. Log the start and end of major operations. Never log sensitive data like passwords or API keys.
