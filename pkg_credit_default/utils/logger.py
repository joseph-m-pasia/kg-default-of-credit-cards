# logger.py
import logging
import sys

# Create a reusable logger
logger = logging.getLogger("house_prices")  # project-wide name
logger.setLevel(logging.INFO)  # default level, can change to DEBUG

# Prevent adding multiple handlers if logger is imported multiple times
if not logger.hasHandlers():
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)  # same level as logger
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)
