# -*- coding: utf-8 -*-
"""日志记录模块"""

import logging
import sys
import os
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler

PROJECT_ROOT = Path(__file__).parent.parent
LOG_DIR = PROJECT_ROOT / "logs"

LOG_DIR.mkdir(exist_ok=True)


class PrintRedirect:
    """重定向print输出到文件和终端"""
    
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'a', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def isatty(self):
        return self.terminal.isatty()


def setup_print_redirect():
    """设置print输出重定向"""
    log_filename = datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.log'
    log_filepath = LOG_DIR / log_filename
    
    # 重定向 stdout
    sys.stdout = PrintRedirect(log_filepath)
    
    # 重定向 stderr
    sys.stderr = PrintRedirect(log_filepath)
    
    print(f"[日志重定向] 已设置输出重定向到: {log_filepath}")


def init_logger(name: str = "subtitle_translation", log_to_file: bool = True):
    """初始化日志记录器

    Args:
        name: 日志记录器名称
        log_to_file: 是否同时写入文件
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_to_file:
        log_filename = datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.log'
        log_filepath = LOG_DIR / log_filename

        file_handler = RotatingFileHandler(
            log_filepath,
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "subtitle_translation"):
    """获取日志记录器实例（已初始化的）"""
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        init_logger(name)
    return logger


def log_with_timestamp(logger, message: str, level: str = "info"):
    """带时间戳的日志记录

    Args:
        logger: 日志记录器实例
        message: 日志消息
        level: 日志级别 (debug/info/warning/error)
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    formatted_message = f"[{timestamp}] {message}"

    if level == "debug":
        logger.debug(formatted_message)
    elif level == "info":
        logger.info(formatted_message)
    elif level == "warning":
        logger.warning(formatted_message)
    elif level == "error":
        logger.error(formatted_message)
    else:
        logger.info(formatted_message)
