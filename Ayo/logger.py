import logging
import os
import sys
import threading
from typing import Any, Dict, Optional


class AyoLogger:

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",  # cyan
        "INFO": "\033[32m",  # green
        "WARNING": "\033[33m",  # yellow
        "ERROR": "\033[31m",  # red
        "CRITICAL": "\033[35m",  # purple
        "RESET": "\033[0m",  # reset
    }

    # singleton instance
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(AyoLogger, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(
        self,
        name: str = "ayo",
        level: str = "INFO",
        log_file: Optional[str] = None,
        use_colors: bool = True,
        log_format: Optional[str] = None,
    ):
        """
        initialize the logger

        Args:
            name: logger name
            level: logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: log file path, if None, only output to console
            use_colors: whether to use colors in console output
            log_format: custom log format, if None, use the default format
        """
        if self._initialized:
            return

        self.name = name
        self.level = getattr(logging, level.upper())
        self.log_file = log_file
        self.use_colors = use_colors

        # create the logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.level)
        self.logger.handlers = []  # clear the existing handlers

        # set the default log format
        if log_format is None:
            self.log_format = "%(asctime)s - %(levelname)s - %(name)s - [%(filename)s:%(lineno)d] - %(message)s"
        else:
            self.log_format = log_format

        # create the formatter
        self.formatter = logging.Formatter(self.log_format, datefmt="%Y-%m-%d %H:%M:%S")

        # add the console handler
        self._add_console_handler()

        # if the log file is specified, add the file handler
        if self.log_file:
            self._add_file_handler()

        self._initialized = True

    def _add_console_handler(self):
        """add the console handler"""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.level)

        if self.use_colors:
            # use the colored formatter
            colored_formatter = self._get_colored_formatter()
            console_handler.setFormatter(colored_formatter)
        else:
            console_handler.setFormatter(self.formatter)

        self.logger.addHandler(console_handler)

    def _add_file_handler(self):
        """add the file handler"""
        # ensure the log directory exists
        log_dir = os.path.dirname(self.log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(self.level)
        file_handler.setFormatter(self.formatter)
        self.logger.addHandler(file_handler)

    def _get_colored_formatter(self):
        """create the colored formatter"""

        class ColoredFormatter(logging.Formatter):
            def __init__(self, fmt, datefmt=None):
                super().__init__(fmt, datefmt)
                self.colors = AyoLogger.COLORS

            def format(self, record):
                levelname = record.levelname
                if levelname in self.colors:
                    record.levelname = (
                        f"{self.colors[levelname]}{levelname}{self.colors['RESET']}"
                    )
                    record.msg = f"{self.colors[levelname.upper()]}{record.msg}{self.colors['RESET']}"
                return super().format(record)

        return ColoredFormatter(self.log_format, datefmt="%Y-%m-%d %H:%M:%S")

    def set_level(self, level: str):
        """set the logging level"""
        level_upper = level.upper()
        if hasattr(logging, level_upper):
            self.level = getattr(logging, level_upper)
            self.logger.setLevel(self.level)
            for handler in self.logger.handlers:
                handler.setLevel(self.level)

    def debug(self, msg: str, *args, **kwargs):
        """record the DEBUG level log"""
        kwargs.setdefault("stacklevel", 2)
        self.logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs):
        """record the INFO level log"""
        kwargs.setdefault("stacklevel", 2)
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs):
        """record the WARNING level log"""
        kwargs.setdefault("stacklevel", 2)
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs):
        """record the ERROR level log"""
        kwargs.setdefault("stacklevel", 2)
        self.logger.error(msg, *args, **kwargs)

    def critical(self, msg: str, *args, **kwargs):
        """record the CRITICAL level log"""
        kwargs.setdefault("stacklevel", 2)
        self.logger.critical(msg, *args, **kwargs)

    def exception(self, msg: str, *args, **kwargs):
        """record the exception information"""
        kwargs.setdefault("stacklevel", 2)
        self.logger.exception(msg, *args, **kwargs)

    def log_dict(self, level: str, data: Dict[str, Any], prefix: str = ""):
        """record the dictionary data"""
        level_method = getattr(self.logger, level.lower())
        for key, value in data.items():
            if prefix:
                key = f"{prefix}.{key}"
            if isinstance(value, dict):
                self.log_dict(level, value, key)
            else:
                level_method(f"{key}: {value}")


def get_logger(
    name: str = "ayo",
    level: str = "INFO",
    log_file: Optional[str] = None,
    use_colors: bool = True,
) -> AyoLogger:
    """
    get the Ayo logger instance

    Args:
        name: logger name
        level: logging level
        log_file: log file path
        use_colors: whether to use colored output

    Returns:
        AyoLogger instance
    """
    return AyoLogger(name=name, level=level, log_file=log_file, use_colors=use_colors)


GLOBAL_INFO_LEVEL = os.environ.get("AYO_INFO_LEVEL", "INFO")

default_logger = AyoLogger(level=GLOBAL_INFO_LEVEL)

debug = default_logger.debug
info = default_logger.info
warning = default_logger.warning
error = default_logger.error
critical = default_logger.critical
exception = default_logger.exception
