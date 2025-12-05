import logging

from helpers import logging_config


def test_set_global_log_level_updates_finefoundry_loggers(tmp_path, monkeypatch):
    # Redirect log files to a temporary directory for this test
    monkeypatch.setattr(logging_config, "LOG_DIR", tmp_path)

    # Create a couple of finefoundry loggers
    logger1 = logging_config.get_logger("finefoundry.test1")
    logger2 = logging_config.get_logger("finefoundry.test2")

    logging_config.set_global_log_level(logging.DEBUG)

    assert logger1.level == logging.DEBUG
    assert logger2.level == logging.DEBUG

    for h in logger1.handlers:
        assert h.level == logging.DEBUG
    for h in logger2.handlers:
        assert h.level == logging.DEBUG
