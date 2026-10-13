import logging
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "input"
OUTPUT_DIR = ROOT / "output"


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def get_logger(log_path: Path) -> logging.Logger:
    """標準出力とファイルの両方に出すロガー"""
    logger = logging.getLogger(str(log_path))
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(message)s", "%Y-%m-%d %H:%M:%S")
    for handler in (logging.StreamHandler(), logging.FileHandler(log_path, mode="w", encoding="utf-8")):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


@contextmanager
def timer(name: str, logger: logging.Logger):
    t0 = time.time()
    yield
    logger.info(f"[{name}] {time.time() - t0:.1f}s")


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))
