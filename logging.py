"""
Logging utilities for training and analysis.
"""

import logging
from pathlib import Path
import json
from datetime import datetime


def setup_logger(name, log_file, level=logging.INFO):
    """
    Set up logger with file and console handlers.
    
    Args:
        name (str): Logger name
        log_file (str): Path to log file
        level: Logging level
        
    Returns:
        logging.Logger: Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    
    # File handler
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def log_experiment_config(config, output_dir):
    """
    Save experiment configuration to JSON.
    
    Args:
        config (dict): Configuration dictionary
        output_dir (str): Output directory
    """
    output_path = Path(output_dir) / 'config.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Add timestamp
    config['timestamp'] = datetime.now().isoformat()
    
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✓ Configuration saved to {output_path}")