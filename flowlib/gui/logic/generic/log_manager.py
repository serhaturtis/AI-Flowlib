"""
Log Manager for GUI application.

Provides centralized logging management and configuration.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


class LogManager:
    """Singleton log manager for the GUI application."""
    
    _instance: Optional['LogManager'] = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def get_instance(cls) -> 'LogManager':
        """Get the singleton instance of LogManager."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.setup_logging()
            self._initialized = True
    
    def setup_logging(self):
        """Set up logging configuration."""
        # Create logs directory
        log_dir = Path.home() / '.flowlib' / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create log file with timestamp
        timestamp = datetime.now().strftime('%Y%m%d')
        log_file = log_dir / f'gui_{timestamp}.log'
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        # Set specific loggers to appropriate levels
        logging.getLogger('PySide6').setLevel(logging.WARNING)
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('requests').setLevel(logging.WARNING)
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("LogManager initialized")
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger with the specified name."""
        return logging.getLogger(name)
    
    def set_level(self, level: int):
        """Set the logging level for all handlers."""
        root_logger = logging.getLogger()
        root_logger.setLevel(level)
        for handler in root_logger.handlers:
            handler.setLevel(level)
        self.logger.info(f"Logging level set to {logging.getLevelName(level)}")
    
    def add_file_handler(self, filename: str, level: int = logging.INFO):
        """Add an additional file handler."""
        log_dir = Path.home() / '.flowlib' / 'logs'
        log_file = log_dir / filename
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)
        self.logger.info(f"Added file handler: {log_file}")
    
    def shutdown(self):
        """Shutdown logging and cleanup."""
        self.logger.info("LogManager shutting down")
        logging.shutdown()