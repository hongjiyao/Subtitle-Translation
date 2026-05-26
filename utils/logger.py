# -*- coding: utf-8 -*-
"""日志记录模块"""

import sys
from pathlib import Path
from datetime import datetime

try:
    from config import PROJECT_ROOT as _PROJECT_ROOT
    PROJECT_ROOT = Path(_PROJECT_ROOT)
except ImportError:
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

    def close(self):
        if hasattr(self, 'log') and self.log and not self.log.closed:
            self.log.close()

    def __del__(self):
        self.close()


def setup_print_redirect():
    """设置print输出重定向"""
    log_filename = datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.log'
    log_filepath = LOG_DIR / log_filename
    
    # 重定向 stdout
    sys.stdout = PrintRedirect(log_filepath)
    
    # 重定向 stderr
    sys.stderr = PrintRedirect(log_filepath)
    
    print(f"[日志重定向] 已设置输出重定向到: {log_filepath}")
