import yaml
import logging
import typing as Dict


def load_config(file_path: str) -> Dict:
    with open(file_path, "r") as f:
        config = yaml.load(f)
    return config


def make_logger(name: str, level = logging.DEBUG) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level=level)
    return logger
