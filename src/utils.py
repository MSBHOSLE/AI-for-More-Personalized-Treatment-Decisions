"""
Utility functions for causal-medical-ai.
Handles configuration loading, reproducibility seeding, and logging.
"""

import os
import random
import logging
import yaml
import numpy as np
import pandas as pd


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across standard libraries."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_yaml(path: str) -> dict:
    """Load configuration dictionary from YAML file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def setup_logger(name: str = "causal_medical_ai", level: int = logging.INFO) -> logging.Logger:
    """Configure structured logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    return logger


def ensure_dir(dir_path: str) -> str:
    """Ensure directory exists, create if missing."""
    os.makedirs(dir_path, exist_ok=True)
    return dir_path


def df_to_markdown(df: pd.DataFrame) -> str:
    """Format DataFrame as a clean GitHub-flavored markdown table without requiring tabulate."""
    if df is None or len(df) == 0:
        return ""
    headers = [str(c) for c in df.columns]
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    data_lines = []
    for _, row in df.iterrows():
        formatted_vals = []
        for v in row:
            if isinstance(v, float):
                formatted_vals.append(f"{v:.4f}")
            else:
                formatted_vals.append(str(v))
        data_lines.append("| " + " | ".join(formatted_vals) + " |")
    return "\n".join([header_line, separator_line] + data_lines)

