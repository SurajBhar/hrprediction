import logging
import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import logger


def test_logger_writes_to_file(tmp_path, monkeypatch):
    # Point DEFAULT_LOG_FILE at a temp file
    test_log = tmp_path / "test.log"
    monkeypatch.setattr(logger, "DEFAULT_LOG_FILE", test_log)

    # Ensure a clean logger state
    name = "tests.logger"
    if name in logging.root.manager.loggerDict:
        del logging.root.manager.loggerDict[name]

    # Get logger and emit a message
    log = logger.get_logger(name, level=logging.INFO)
    message = "Test log entry"
    log.info(message)

    # Flush and close handlers so data is written
    for h in log.handlers:
        h.flush()
        h.close()

    # Assert the file was created and contains our message
    assert test_log.exists(), "Log file was not created"
    content = test_log.read_text(encoding="utf-8")
    assert message in content, f"Expected '{message}' in log file, got:\n{content}"
