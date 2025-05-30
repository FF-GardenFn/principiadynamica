"""
Constitutional Dynamics Configuration Module

This module provides configuration loading and management for the Constitutional Dynamics package.
"""

import os
import yaml
import logging
import logging.config
from typing import Dict, Any, Optional

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "defaults.yaml")
DEFAULT_LOGGING_PATH = os.path.join(os.path.dirname(__file__), "logging.yaml")

logger = logging.getLogger("constitutional_dynamics.cfg")

def _update_dict(d: Dict, u: Dict) -> Dict:
    """
    Recursively update a dictionary.

    Args:
        d: Dictionary to update
        u: Dictionary with updates

    Returns:
        Updated dictionary
    """
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = _update_dict(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def get_default_config() -> Dict[str, Any]:
    """
    Get the default configuration.

    Returns:
        Default configuration dictionary
    """
    # Default configuration
    return {
        "memory": {
            "decay_rate": 0.2,
            "window_size": 10
        },
        "alignment": {
            "similarity_threshold": 0.7
        },
        "optimizer": {
            "lambda_weight": 0.35,
            "flow_constraint_strength": 5.0,
            "quantum_num_reads": 1000
        },
        "visualization": {
            "sparkline_width": 60,
            "use_rich": True
        },
        "live": {
            "interval": 1.0,
            "gauss_sigma": 2.0,
            "max_queue": 512
        }
    }

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from file and merge with default configuration.

    Args:
        config_path: Path to configuration file (defaults to cfg/defaults.yaml)

    Returns:
        Configuration dictionary
    """
    # Get default configuration
    config = get_default_config()

    # If no config path provided, use default
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    # If config file exists, load and merge with default
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f)
                if file_config:
                    # Update default config with file config
                    _update_dict(config, file_config)
                    logger.debug(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.warning(f"Error loading configuration from {config_path}: {e}")
            logger.warning("Using default configuration")
    else:
        logger.warning(f"Configuration file not found: {config_path}")
        logger.warning("Using default configuration")

    return config

def get_default_logging_config() -> Dict[str, Any]:
    """
    Get the default logging configuration.

    Returns:
        Default logging configuration dictionary
    """
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "standard",
                "stream": "ext://sys.stdout"
            },
            "file": {
                "class": "logging.FileHandler",
                "level": "DEBUG",
                "formatter": "standard",
                "filename": "constitutional_dynamics.log",
                "mode": "a"
            }
        },
        "loggers": {
            "constitutional_dynamics": {
                "level": "INFO",
                "handlers": ["console", "file"],
                "propagate": False
            }
        },
        "root": {
            "level": "WARNING",
            "handlers": ["console"]
        }
    }

def load_logging_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load logging configuration from file.

    Args:
        config_path: Path to logging configuration file (defaults to cfg/logging.yaml)

    Returns:
        Logging configuration dictionary
    """
    # Get default logging configuration
    config = get_default_logging_config()

    # If no config path provided, use default
    if config_path is None:
        config_path = DEFAULT_LOGGING_PATH

    # If config file exists, load and merge with default
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f)
                if file_config:
                    # Update default config with file config
                    _update_dict(config, file_config)
                    logger.debug(f"Loaded logging configuration from {config_path}")
        except Exception as e:
            logger.warning(f"Error loading logging configuration from {config_path}: {e}")
            logger.warning("Using default logging configuration")
    else:
        logger.warning(f"Logging configuration file not found: {config_path}")
        logger.warning("Using default logging configuration")

    return config

def configure_logging(config_path: Optional[str] = None) -> None:
    """
    Configure logging using the specified configuration file.

    Args:
        config_path: Path to logging configuration file (defaults to cfg/logging.yaml)
    """
    config = load_logging_config(config_path)

    try:
        logging.config.dictConfig(config)
        logger.debug("Logging configured successfully")
    except Exception as e:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        logger.warning(f"Error configuring logging: {e}")
        logger.warning("Using basic logging configuration")

__all__ = [
    "load_config",
    "get_default_config",
    "load_logging_config",
    "get_default_logging_config",
    "configure_logging",
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_LOGGING_PATH"
]