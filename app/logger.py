import logging


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(filename)s - %(levelname)s - %(message)s',
)

    file_handler = logging.FileHandler('app.log')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger