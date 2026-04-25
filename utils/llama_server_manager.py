# -*- coding: utf-8 -*-
"""
llama-server 进程管理模块
负责 llama-server 进程的启动、停止、健康检测和自动重启
"""

import os
import time
import subprocess
import requests
import glob
import signal
from typing import Optional, Dict, Any

from config import config, PROJECT_ROOT


class LlamaServerManager:
    def __init__(self, system_prompt=None, port=None):
        self.host = config.get('llama_server_host', '127.0.0.1')
        self.port = port or config.get('llama_server_port', 8080)
        self.context_size = config.get('llama_server_context_size', 8192)
        self.threads = config.get('llama_server_threads', 8)
        self.system_prompt = system_prompt

        self.process: Optional[subprocess.Popen] = None
        self.server_path: Optional[str] = None
        self.model_path: Optional[str] = None
        self.fail_count = 0
        self.max_failures = 3

        self._find_server_path()
        self._find_model_path()

    def _find_server_path(self):
        possible_paths = [
            os.path.join(PROJECT_ROOT, "llama_cpp", "llama-server.exe"),
            os.path.join(PROJECT_ROOT, "llama_cpp", "build", "llama-server.exe"),
        ]
        for path in possible_paths:
            if os.path.exists(path):
                self.server_path = path
                return

        for exe_name in ["llama-server.exe"]:
            search_pattern = os.path.join(PROJECT_ROOT, "llama_cpp", "**", exe_name)
            found = glob.glob(search_pattern, recursive=True)
            if found:
                self.server_path = found[0]
                return

        self.server_path = None

    def _find_model_path(self):
        from config import MODEL_CACHE_DIR
        quantization = config.get('translator_quantization', 'auto')

        # 打印 MODEL_CACHE_DIR 路径
        print(f"[llama-server] 模型缓存目录: {MODEL_CACHE_DIR}")

        model_dirs = [
            os.path.join(MODEL_CACHE_DIR, "tencent--HY-MT1.5-7B-GGUF"),
            os.path.join(MODEL_CACHE_DIR, "HY-MT1.5-7B-GGUF"),
        ]

        for model_dir in model_dirs:
            print(f"[llama-server] 检查模型目录: {model_dir}")
            if os.path.exists(model_dir):
                print(f"[llama-server] 找到模型目录: {model_dir}")
                gguf_files = glob.glob(os.path.join(model_dir, "*.gguf"))
                print(f"[llama-server] 找到 GGUF 文件: {gguf_files}")
                if gguf_files:
                    if quantization != "auto":
                        print(f"[llama-server] 按量化版本查找: {quantization}")
                        for f in gguf_files:
                            if quantization in f:
                                self.model_path = f
                                print(f"[llama-server] 找到匹配的模型: {f}")
                                return
                    print("[llama-server] 按优先顺序查找模型")
                    for preferred in ["Q2_K", "Q3_K_S", "Q3_K_M", "Q4_0", "Q4_K_S", "Q4_K_M", "Q5_0", "Q5_K_S", "Q5_K_M", "Q6_K", "Q8_0"]:
                        for f in gguf_files:
                            if preferred in f:
                                self.model_path = f
                                print(f"[llama-server] 找到优先模型: {f}")
                                return
                    self.model_path = gguf_files[0]
                    print(f"[llama-server] 使用第一个模型: {gguf_files[0]}")
                    return

        search_pattern = os.path.join(MODEL_CACHE_DIR, "**", "*.gguf")
        print(f"[llama-server] 递归搜索模型: {search_pattern}")
        found = glob.glob(search_pattern, recursive=True)
        print(f"[llama-server] 递归搜索结果: {found}")
        if found:
            if quantization != "auto":
                print(f"[llama-server] 按量化版本查找: {quantization}")
                for f in found:
                    if quantization in f:
                        self.model_path = f
                        print(f"[llama-server] 找到匹配的模型: {f}")
                        return
            smallest_file = min(found, key=os.path.getsize)
            self.model_path = smallest_file
            print(f"[llama-server] 使用最小模型: {smallest_file}")
            return

        self.model_path = None
        print("[llama-server] 未找到模型文件")

    def is_server_running(self) -> bool:
        if self.process is None:
            return False
        if self.process.poll() is not None:
            return False
        return self._health_check()

    def _health_check(self) -> bool:
        url = f"http://{self.host}:{self.port}/health"
        try:
            response = requests.get(url, timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def start_server(self) -> bool:
        if self.is_server_running():
            return True

        if self.server_path is None:
            print(f"[llama-server] 错误: 找不到 llama-server.exe")
            return False

        if self.model_path is None:
            print(f"[llama-server] 错误: 找不到 GGUF 模型文件")
            return False

        self.stop_server()

        cmd = [
            self.server_path,
            "-m", self.model_path,
            "--host", self.host,
            "--port", str(self.port),
            "-c", str(self.context_size),
            "-b", "2048",  # 设置批处理大小为2048
            "-t", str(self.threads),
            "-ngl", "99",
            "--no-mmap",  # 关闭内存映射
        ]
        
        # 添加系统提示词（如果提供）
        if self.system_prompt:
            # 使用环境变量设置系统提示词，避免命令行参数解析问题
            os.environ['LLAMA_SYSTEM_PROMPT'] = self.system_prompt

        try:
            print(f"[llama-server] 启动命令: {' '.join(cmd)}")
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                errors='ignore'
            )
            time.sleep(3)

            if self.process.poll() is not None:
                stdout, stderr = self.process.communicate()
                print(f"[llama-server] 启动失败，退出码: {self.process.returncode}")
                print(f"[llama-server] 标准输出: {stdout[:500]}")
                print(f"[llama-server] 标准错误: {stderr[:500]}")
                return False

            for _ in range(10):
                if self._health_check():
                    self.fail_count = 0
                    print(f"[llama-server] 服务器已启动: {self.host}:{self.port}")
                    return True
                time.sleep(1)

            print(f"[llama-server] 启动超时")
            return False

        except Exception as e:
            print(f"[llama-server] 启动异常: {e}")
            return False

    def stop_server(self):
        if self.process is None:
            return

        try:
            if os.name == 'nt':
                self.process.terminate()
            else:
                self.process.send_signal(signal.SIGTERM)
            
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                if os.name == 'nt':
                    self.process.kill()
                else:
                    self.process.send_signal(signal.SIGKILL)
                self.process.wait()
        except Exception:
            pass

        self.process = None

    def ensure_server_running(self) -> bool:
        if self.is_server_running():
            self.fail_count = 0
            return True

        print(f"[llama-server] 服务器未运行，正在启动...")
        
        if not self.start_server():
            self.fail_count += 1
            if self.fail_count >= self.max_failures:
                print(f"[llama-server] 服务器连续启动失败 {self.fail_count} 次，尝试强制重启")
                self.fail_count = 0
            return False

        self.fail_count = 0
        return True

    def reset_session(self) -> bool:
        """
        重置会话状态
        通过重启服务器来确保全新的会话环境
        
        Returns:
            bool: 重置是否成功
        """
        print(f"[llama-server] 正在重置会话...")
        start_time = time.time()
        
        # 停止并重启服务器
        self.stop_server()
        success = self.start_server()
        
        end_time = time.time()
        execution_time = (end_time - start_time) * 1000
        print(f"[llama-server] 会话重置 {'成功' if success else '失败'}，耗时: {execution_time:.2f}ms")
        
        return success

    def send_request(self, prompt: str, **kwargs) -> Optional[str]:
        if not self.ensure_server_running():
            return None

        url = f"http://{self.host}:{self.port}/completion"

        params = {
            "prompt": prompt,
            "n_predict": kwargs.get("n_predict", 256),
            "temperature": kwargs.get("temperature", 0.0),
            "top_k": kwargs.get("top_k", 20),
            "top_p": kwargs.get("top_p", 0.6),
            "repeat_penalty": kwargs.get("repeat_penalty", 1.05),
            "stop": kwargs.get("stop", []),
        }

        try:
            response = requests.post(
                url,
                json=params,
                timeout=kwargs.get("timeout", 120)
            )
            
            if response.status_code == 200:
                self.fail_count = 0
                return response.json().get("content", "")
            else:
                self.fail_count += 1
                print(f"[llama-server] 请求失败: HTTP {response.status_code}")
                return None

        except requests.exceptions.Timeout:
            self.fail_count += 1
            print(f"[llama-server] 请求超时")
            if self.fail_count >= self.max_failures:
                print(f"[llama-server] 检测到 {self.fail_count} 次连续失败，尝试重启服务器")
                self.stop_server()
                self.fail_count = 0
            return None

        except requests.exceptions.RequestException as e:
            self.fail_count += 1
            print(f"[llama-server] 请求异常: {e}")
            return None




__all__ = ['LlamaServerManager']
