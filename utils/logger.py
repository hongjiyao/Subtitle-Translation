# -*- coding: utf-8 -*-
"""日志记录模块"""

import atexit
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
        try:
            self.close()
        except AttributeError:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


def setup_print_redirect():
    log_filename = datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.log'
    log_filepath = LOG_DIR / log_filename

    sys.stdout = PrintRedirect(log_filepath)
    sys.stderr = PrintRedirect(log_filepath)

    def _cleanup():
        if isinstance(sys.stdout, PrintRedirect):
            sys.stdout.close()
        if isinstance(sys.stderr, PrintRedirect):
            sys.stderr.close()

    atexit.register(_cleanup)

    print(f"[日志重定向] 已设置输出重定向到: {log_filepath}")
