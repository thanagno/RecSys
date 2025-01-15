import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging(logger_root_dir: Path, log_level: str = 'INFO', mode: str = 'w'):
    """
    Sets up logging for the application with a rotating file handler and console output.

    Args:
        logger_root_dir: The directory where the log files will be stored.
        log_level: The log level to capture. Default is 'INFO'.
        mode: the mode of the log file. Can be 'a' for appending the logs or 'w' for overwriting them
    """
    # Ensure the loggers directory exists
    if not logger_root_dir.exists():
        logger_root_dir.mkdir(parents=True, exist_ok=True)

    # Path to the log file
    log_file = logger_root_dir / 'app.log'

    # Define the logging level
    log_level = getattr(logging, log_level.upper(), log_level)

    # Create the root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)  # Set the logging level (INFO or higher)

    # File handler with rotation (1MB max file size, keep 5 backups)
    file_handler = RotatingFileHandler(log_file, maxBytes=1 * 1024 * 1024, backupCount=5, mode=mode)
    file_handler.setLevel(log_level)  # Set the log level for the file handler

    # Create a formatter and add it to the file handler
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s [%(filename)s:%(lineno)d - %(funcName)s] - %(message)s',
        datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)

    # Console handler: display INFO and above messages to console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # Only show INFO and above on the console
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
