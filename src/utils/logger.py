"""
Logging utilities for training and active learning.
"""

import logging
import sys
import os


def setup_logger(name: str = "brain_ct_seg", log_file: str = "outputs/training.log") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger

    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
