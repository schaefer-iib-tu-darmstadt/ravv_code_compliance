"""Logging setup and log-level helper."""
import logging
import sys


def set_logging_level(level: str = 'info'):
    if level.lower() == 'debug':
        logging_level = logging.DEBUG
    elif level.lower() == 'info':
        logging_level = logging.INFO
    elif level.lower() == 'warning':
        logging_level = logging.WARNING
    elif level.lower() == 'error':
        logging_level = logging.ERROR
    elif level.lower() == 'critical':
        logging_level = logging.CRITICAL
    logging.basicConfig(stream=sys.stdout, level=logging_level)
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
