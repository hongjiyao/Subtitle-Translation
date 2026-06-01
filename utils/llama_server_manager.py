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

from config import config, ServerParams, PROJECT_ROOT


class LlamaServerManager:
    def __init__(self, port=None, server_params: ServerParams = None):
        if server_params is None:
            server_params = ServerParams()
        self._server_params = server_params
        self.host = server_params.host
        self.port = port or server_params.port
        self.context_size = server_params.ctx_size
        self.threads = server_params.threads

        self.process: Optional[subprocess.Popen] = None
        self.pid = None
        self.server_path: Optional[str] = None
        self.model_path: Optional[str] = None
        self.fail_count = 0
        self.max_failures = 3
        self._log_file = None

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
        print(f"[llama-server] 未找到 llama-server 可执行文件")

    @classmethod
    def find_model_path(cls, model_name):
        from config import MODEL_CACHE_DIR
        translator = config.get('translator', 'tencent/HY-MT1.5-1.8B-GGUF-Q8_0')

        quantization = None
        for q in ["Q8_0", "Q6_K", "Q5_K_M", "Q5_K_S", "Q5_0", "Q4_K_M", "Q4_K_S", "Q4_0", "Q3_K_M", "Q3_K_S", "Q2_K"]:
            if translator.endswith("-" + q):
                quantization = q
                translator_repo = translator[:-(len(q) + 1)]
                break
        if quantization is None:
            translator_repo = translator

        print(f"[llama-server] 模型缓存目录: {MODEL_CACHE_DIR}")
        print(f"[llama-server] 当前翻译模型: {translator}")
        print(f"[llama-server] 仓库ID: {translator_repo}, 量化版本: {quantization}")

        translator_dir_name = translator_repo.replace("/", "--")
        model_dirs = [
            os.path.join(MODEL_CACHE_DIR, translator_dir_name),
        ]

        if translator_repo == "tencent/HY-MT1.5-1.8B-GGUF":
            model_dirs.append(os.path.join(MODEL_CACHE_DIR, "HY-MT1.5-1.8B-GGUF"))
        elif translator_repo == "SakuraLLM/Sakura-7B-Qwen2.5-v1.0-GGUF":
            model_dirs.append(os.path.join(MODEL_CACHE_DIR, "Sakura-7B-Qwen2.5-v1.0-GGUF"))

        for model_dir in model_dirs:
            print(f"[llama-server] 检查模型目录: {model_dir}")
            if os.path.exists(model_dir):
                print(f"[llama-server] 找到模型目录: {model_dir}")
                gguf_files = glob.glob(os.path.join(model_dir, "*.gguf"))
                print(f"[llama-server] 找到 GGUF 文件: {gguf_files}")
                if gguf_files:
                    if quantization:
                        print(f"[llama-server] 按量化版本查找: {quantization}")
                        for f in gguf_files:
                            if quantization in f:
                                print(f"[llama-server] 找到匹配的模型: {f}")
                                return f
                    print("[llama-server] 按优先顺序查找模型")
                    for preferred in ["Q2_K", "Q3_K_S", "Q3_K_M", "Q4_0", "Q4_K_S", "Q4_K_M", "Q5_0", "Q5_K_S", "Q5_K_M", "Q6_K", "Q8_0"]:
                        for f in gguf_files:
                            if preferred in f:
                                print(f"[llama-server] 找到优先模型: {f}")
                                return f
                    print(f"[llama-server] 使用第一个模型: {gguf_files[0]}")
                    return gguf_files[0]

        search_pattern = os.path.join(MODEL_CACHE_DIR, "**", "*.gguf")
        print(f"[llama-server] 递归搜索模型: {search_pattern}")
        found = glob.glob(search_pattern, recursive=True)
        print(f"[llama-server] 递归搜索结果: {found}")
        if found:
            if quantization:
                print(f"[llama-server] 按量化版本查找: {quantization}")
                for f in found:
                    if quantization in f:
                        print(f"[llama-server] 找到匹配的模型: {f}")
                        return f
            smallest_file = min(found, key=os.path.getsize)
            print(f"[llama-server] 使用最小模型: {smallest_file}")
            return smallest_file

        print("[llama-server] 未找到模型文件")
        return None

    def _find_model_path(self):
        self.model_path = self.find_model_path(None)

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
            print(f"[llama-server] 健康检查失败: {self.host}:{self.port}")
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

        ngl = self._server_params.ngl
        batch_size = self._server_params.batch_size
        parallel_slots = self._server_params.parallel_slots
        cmd = [
            self.server_path,
            "-m", self.model_path,
            "--host", self.host,
            "--port", str(self.port),
            "-c", str(self.context_size),
            "-b", str(batch_size),
            "-t", str(self.threads),
            "-ngl", str(ngl),
            "-np", str(parallel_slots),
            "--flash-attn", "auto",
            "--no-mmap",
        ]
        
        try:
            print(f"[llama-server] 启动命令: {' '.join(cmd)}")
            log_dir = os.path.join(PROJECT_ROOT, "logs")
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, "llama_server.log")
            try:
                self._log_file = open(log_path, "w", encoding="utf-8")
            except Exception:
                self._log_file = None
            try:
                self.process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=self._log_file if self._log_file else subprocess.DEVNULL,
                )
            except Exception:
                if self._log_file is not None:
                    self._log_file.close()
                    self._log_file = None
                raise
            self.pid = self.process.pid
            time.sleep(0.5)

            if self.process.poll() is not None:
                print(f"[llama-server] 启动失败，退出码: {self.process.returncode}")
                return False

            for _ in range(15):
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
                subprocess.run(['taskkill', '/F', '/T', '/PID', str(self.process.pid)],
                               capture_output=True, timeout=5)
            else:
                self.process.send_signal(signal.SIGTERM)

            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print(f"[llama-server] 停止服务器超时，强制终止进程")
                if os.name != 'nt':
                    self.process.send_signal(signal.SIGKILL)
                    self.process.wait()
        except Exception as e:
            print(f"[llama-server] 停止服务器异常: {e}")

        self.process = None
        self.pid = None

        if self._log_file is not None:
            try:
                self._log_file.close()
            except Exception:
                pass
            self._log_file = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_server()
        return False

    def __del__(self):
        try:
            self.stop_server()
        except Exception:
            pass
        if self._log_file is not None:
            try:
                self._log_file.close()
            except Exception:
                pass
            self._log_file = None

    def ensure_server_running(self) -> bool:
        if self.is_server_running():
            self.fail_count = 0
            return True

        print(f"[llama-server] 服务器未运行，正在启动...")
        
        if not self.start_server():
            self.fail_count += 1
            print(f"[llama-server] 服务器启动失败 (第{self.fail_count}次)")
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

    def send_chat_request(self, messages: list, **kwargs) -> Optional[str]:
        if not self.ensure_server_running():
            return None

        url = f"http://{self.host}:{self.port}/v1/chat/completions"

        params = {
            "messages": messages,
            "n_predict": kwargs.get("n_predict", 256),
            "temperature": kwargs.get("temperature", 0.0),
            "top_k": kwargs.get("top_k", 20),
            "top_p": kwargs.get("top_p", 0.6),
            "repeat_penalty": kwargs.get("repeat_penalty", 1.05),
            "stop": kwargs.get("stop", []),
            "cache_prompt": True,
        }

        try:
            response = requests.post(
                url,
                json=params,
                timeout=kwargs.get("timeout", 120)
            )

            if response.status_code == 200:
                self.fail_count = 0
                data = response.json()
                choices = data.get("choices", [])
                if choices:
                    message = choices[0].get("message", {})
                    return message.get("content", "")
                print(f"[llama-server] Chat请求返回空choices")
                return ""
            else:
                self.fail_count += 1
                try:
                    error_body = response.text[:500]
                except Exception:
                    print(f"[llama-server] 获取错误响应体失败")
                    error_body = ""
                print(f"[llama-server] Chat请求失败: HTTP {response.status_code}, 响应: {error_body}")
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
