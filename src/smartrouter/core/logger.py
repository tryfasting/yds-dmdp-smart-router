import os
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

def setup_logger(name: str = "smartrouter") -> logging.Logger:
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers if setup multiple times
    if logger.handlers:
        return logger
        
    logger.setLevel(logging.INFO)
    
    # Define logging format
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s:%(filename)s:%(lineno)d] - %(message)s"
    )
    
    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File Handler (Rotate logs if they reach 5MB, keep 5 backups)
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    file_handler = RotatingFileHandler(
        log_dir / "smartrouter.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    logger.info("📡 Custom Logger initialized successfully (Console & File).")
    return logger

# Global default logger
logger = setup_logger()
