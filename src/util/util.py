import logging
from pathlib import Path


def setup_logger(logger: logging.Logger, save_dir: Path | None = None, log_filename: str = "log.log"):
    # Remove base handlers
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Set level for logger
    logger.setLevel(logging.INFO)

    # Create a formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    if save_dir is not None:
        # Create handler
        file_handler = logging.FileHandler(save_dir / log_filename)  # Logs to a file

        # Attach formatter to the handler
        file_handler.setFormatter(formatter)

        # Add handlers to the logger
        root_logger.addHandler(file_handler)

    # Create handler
    console_handler = logging.StreamHandler()  # Logs to console

    # Attach formatter to the handler
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    root_logger.addHandler(console_handler)
