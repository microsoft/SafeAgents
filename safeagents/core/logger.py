import logging
import os
import re
import sys
from typing import Literal

from termcolor import colored

# Configuration from environment variables
LOG_LEVEL = os.getenv('LOG_LEVEL', 'DEBUG').upper()
DEBUG = os.getenv('DEBUG', 'TRUE').lower() in ['true', '1', 'yes']
DISABLE_COLOR = os.getenv('DISABLE_COLOR', 'False').lower() in ['true', '1', 'yes']

if DEBUG:
    LOG_LEVEL = 'DEBUG'

ColorType = Literal[
    'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 
    'light_grey', 'dark_grey', 'light_red', 'light_green', 
    'light_yellow', 'light_blue', 'light_magenta', 'light_cyan', 'white'
]

# Color mapping for different log levels and message types
LOG_COLORS = {
    'DEBUG': 'cyan',
    'INFO': 'green', 
    'WARNING': 'yellow',
    'ERROR': 'red',
    'CRITICAL': 'light_red',
    'ACTION': 'green',
    'OBSERVATION': 'yellow',
    'DETAIL': 'cyan',
}


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from string."""
    pattern = re.compile(r'\x1B\[\d+(;\d+){0,2}m')
    return pattern.sub('', text)


class ColoredFormatter(logging.Formatter):
    """Simple colored formatter for console output."""
    
    def format(self, record: logging.LogRecord) -> str:
        # Get message type from record if available
        msg_type = getattr(record, 'msg_type', record.levelname)
        
        if not DISABLE_COLOR and msg_type in LOG_COLORS:
            # Color the message and timestamp
            colored_msg = colored(record.getMessage(), LOG_COLORS[msg_type])
            colored_time = colored(self.formatTime(record, self.datefmt), LOG_COLORS[msg_type])
            colored_level = colored(record.levelname, LOG_COLORS[msg_type])
            
            if record.levelno >= logging.ERROR or DEBUG:
                # Include file and line info for errors and debug mode
                return f'{colored_time} - {record.name}:{colored_level}: {record.filename}:{record.lineno} - {colored_msg}'
            else:
                return f'{colored_time} - {colored_level} - {colored_msg}'
        else:
            # No color - standard format
            if record.levelno >= logging.ERROR or DEBUG:
                return f'{self.formatTime(record, self.datefmt)} - {record.name}:{record.levelname}: {record.filename}:{record.lineno} - {record.getMessage()}'
            else:
                return f'{self.formatTime(record, self.datefmt)} - {record.levelname} - {record.getMessage()}'


def get_console_handler(log_level: int = logging.INFO) -> logging.StreamHandler:
    """Create and configure console handler with colored output."""
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    
    formatter = ColoredFormatter(datefmt='%H:%M:%S')
    console_handler.setFormatter(formatter)
    
    return console_handler


def get_logger(name: str = 'safeagents') -> logging.Logger:
    """Get a configured logger instance."""
    logger = logging.getLogger(name)
    
    # Set log level
    current_log_level = getattr(logging, LOG_LEVEL, logging.INFO)
    logger.setLevel(current_log_level)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Add console handler
    logger.addHandler(get_console_handler(current_log_level))
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


class LoggerAdapter(logging.LoggerAdapter):
    """Simple adapter to add message types to log records."""
    
    def __init__(self, logger: logging.Logger, extra: dict = None):
        super().__init__(logger, extra or {})
    
    def action(self, msg, *args, **kwargs):
        """Log an action message."""
        self._log_with_type('ACTION', msg, *args, **kwargs)
    
    def observation(self, msg, *args, **kwargs):
        """Log an observation message."""
        self._log_with_type('OBSERVATION', msg, *args, **kwargs)
    
    def detail(self, msg, *args, **kwargs):
        """Log a detail message."""
        self._log_with_type('DETAIL', msg, *args, **kwargs)
    
    def _log_with_type(self, msg_type: str, msg, *args, **kwargs):
        """Log message with specific type."""
        extra = kwargs.get('extra', {})
        extra['msg_type'] = msg_type
        kwargs['extra'] = extra
        self.info(msg, *args, **kwargs)


# Create default logger instance
safeagents_logger = get_logger('safeagents')

# Create adapter for enhanced logging
logger = LoggerAdapter(safeagents_logger)
