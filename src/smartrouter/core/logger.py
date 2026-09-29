import logging


def setup_logger(name: str = "smartrouter") -> logging.Logger:
    """Console-only logger. Input sentences must never be logged."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s - %(message)s"))
    logger.addHandler(handler)
    return logger


logger = setup_logger()
