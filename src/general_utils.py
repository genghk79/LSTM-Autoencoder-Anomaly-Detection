"""Utilities or functions that are useful across different projects can be defined here."""

import logging
import logging.config
import os
import yaml

logger = logging.getLogger(__name__)

def setup_logging(
    logging_config_path="./configs/logging.yaml", 
    default_level=logging.INFO, 
    log_dir="./logs/"
):
    """Set up configuration for logging utilities.

    Parameters
    ----------
    logging_config_path : str, optional
        Path to YAML file containing configuration for Python logger,
        by default "./configs/logging.yaml"
    default_level : 
        logging object, optional, by default logging.INFO
    log_dir : str, optional
        Directory to store log files, by "./logs/"
    """

    try:
        with open(logging_config_path, "rt", encoding="utf-8") as file:
            log_config = yaml.safe_load(file.read())

        # Modify log file paths if log_dir is provided
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            for handler in log_config["handlers"].values():
                if "filename" in handler:
                    # Extract just the filename from the path
                    filename = os.path.basename(handler["filename"])
                    # Create new path with the provided directory
                    handler["filename"] = os.path.join(log_dir, filename)

        logging.config.dictConfig(log_config)

    except Exception as error:
        logging.basicConfig(
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            level=default_level,
        )
        logger.error(error)
        logger.error("Logging config file is not found. Basic config is being used.")

setup_logging()

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file.

    Args:
        config_path: Path to configuration YAML file.

    Returns:
        Configuration dictionary.
    """

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Could not load config {config_path}: {e}")

