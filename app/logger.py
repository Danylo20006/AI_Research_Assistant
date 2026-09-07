import logging

def get_logger(name: str) -> logging.Logger:
    """
    Initializes and returns a configured logger instance.
    """
    logger = logging.getLogger(name)

    # Prevent adding multiple handlers if the logger is imported in multiple modules.
    # This avoids duplicate log lines in the output file.
    if not logger.handlers:
        logger.setLevel(logging.INFO)

        formatter = logging.Formatter(
            '%(asctime)s - %(filename)s - %(levelname)s - %(message)s',
        )

        file_handler = logging.FileHandler('app.log')
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)

    return logger