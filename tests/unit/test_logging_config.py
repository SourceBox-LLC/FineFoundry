from helpers import logging_config


def test_get_logger_returns_consistent_logger():
    logger1 = logging_config.get_logger(__name__)
    logger2 = logging_config.get_logger(__name__)

    assert logger1 is logger2
    assert logger1.name == __name__


def test_get_logger_has_stream_handler():
    logger = logging_config.get_logger("finefoundry.test")
    # At least one handler should be attached, typically a StreamHandler
    assert logger.handlers
