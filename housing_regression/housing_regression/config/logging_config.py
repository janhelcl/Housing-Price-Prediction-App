"""
Configuration of logging througout the whole package
"""
import logging
import logging.handlers
import os
import sys

FORMATTER = logging.Formatter(
    "%(asctime)s — %(name)s — %(levelname)s —" "%(funcName)s:%(lineno)d — %(message)s"
)


LOG_DIR = os.path.join(os.path.dirname(__file__), "..\logs")
LOG_FILE = LOG_DIR + "\logs.log"


if not os.path.exists(LOG_FILE):
    os.makedirs(LOG_DIR)
    open(LOG_FILE, "a").close()


def get_console_handler():
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)
    return console_handler


def get_file_handler():
    file_handler = logging.handlers.TimedRotatingFileHandler(LOG_FILE, when="midnight")
    file_handler.setFormatter(FORMATTER)
    return file_handler


def get_logger(logger_name, level=logging.INFO):
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.addHandler(get_console_handler())
    logger.addHandler(get_file_handler())
    logger.propagate = False
    return logger
