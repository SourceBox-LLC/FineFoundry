import logging

from helpers import logging_config


def test_set_global_log_level_updates_finefoundry_loggers(tmp_path, monkeypatch):
    # Clear any existing handlers to avoid duplicates
    for name in ["finefoundry.test_level1", "finefoundry.test_level2"]:
        logger = logging.getLogger(name)
        logger.handlers.clear()

    # Create a couple of finefoundry loggers
    logger1 = logging_config.get_logger("finefoundry.test_level1")
    logger2 = logging_config.get_logger("finefoundry.test_level2")

    logging_config.set_global_log_level(logging.DEBUG)

    assert logger1.level == logging.DEBUG
    assert logger2.level == logging.DEBUG

    for h in logger1.handlers:
        assert h.level == logging.DEBUG
    for h in logger2.handlers:
        assert h.level == logging.DEBUG
