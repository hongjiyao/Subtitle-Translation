# -*- coding: utf-8 -*-
import os
import time

# 强制使用 UTF-8 编码
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['LC_ALL'] = 'en_US.UTF-8'
os.environ['LANG'] = 'en_US.UTF-8'

import datetime
import gc
import subprocess
import glob

# 设置HF-Mirror作为下载源
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TRANSFORMERS_OFFLINE"] = "1"  # 强制离线模式，禁止自动下载
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.environ["HUGGINGFACE_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

from config import MODEL_CACHE_DIR, PROJECT_ROOT


def timestamp_print(message):
    """带时间戳的打印函数"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


def clear_translator_cache():
    """清空翻译模型缓存以释放内存"""
    gc.collect()
    timestamp_print("[内存管理] 已执行垃圾回收")


class LlamaCppTranslator:
    """使用 llama-completion.exe 运行 GGUF 模型的翻译器 - 单例模式"""
    _instance = None
    _initialized = False

    def __new__(cls, model_path=None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, model_path=None):
        if not self._initialized:
            # 查找 llama-completion.exe 路径（非交互式版本更稳定）
            self.llama_cli_path = self._find_llama_cli()
            if not self.llama_cli_path:
                raise RuntimeError("找不到 llama-completion.exe 或 llama-cli.exe，请确保 llama_cpp 目录存在")

            # 查找 GGUF 模型文件
            self.model_path = model_path or self._find_gguf_model()
            if not self.model_path:
                raise RuntimeError("找不到 GGUF 模型文件，请确保模型已下载")

            # 从配置中获取上下文大小
            from config import config
            self.context_size = config.get('translation_context_size', 10000)
            self.process = None
            self.stdin = None
            self.stdout = None
            self.stderr = None

            timestamp_print(f"[llama.cpp翻译] 翻译器: {self.llama_cli_path}")
            timestamp_print(f"[llama.cpp翻译] 模型: {self.model_path}")
            timestamp_print(f"[llama.cpp翻译] 默认上下文大小: {self.context_size}")
            self._initialized = True

    def start_process(self, context_size=None):
        """启动持久化的 llama 进程 - 使用标准输入/输出实现"""
        # 检查是否已有进程在运行
        if self.process and self.process.poll() is None:
            timestamp_print("[llama.cpp翻译] 服务器进程已经在运行")
            return True

        # 使用配置中的上下文大小作为默认值
        if context_size is None:
            context_size = self.context_size

        max_attempts = 3
        attempt = 0
        
        while attempt < max_attempts:
            attempt += 1
            timestamp_print(f"[llama.cpp翻译] 尝试启动服务器进程 (尝试 {attempt}/{max_attempts})...")
            
            try:
                # 从配置获取翻译参数
                from config import config
                temperature = config.get('translation_temperature', 0.05)
                top_k = config.get('translation_top_k', 40)
                top_p = config.get('translation_top_p', 0.95)
                repetition_penalty = config.get('translation_repetition_penalty', 1.0)
                
                # 构建命令 - 使用简单IO模式
                cmd = [
                    self.llama_cli_path,
                    "-m", self.model_path,
                    "-t", "8",  
                    "-c", str(context_size),
                    "--temp", str(temperature),
                    "--top-p", str(top_p),
                    "--top-k", str(top_k),
                    "--repeat-penalty", str(repetition_penalty),
                    "-ngl", "99",
                    "--simple-io",
                    "--no-display-prompt"
                ]

                timestamp_print(f"[llama.cpp翻译] 启动命令: {' '.join(cmd)}")
                timestamp_print(f"[llama.cpp翻译] 模型路径: {self.model_path}")
                timestamp_print(f"[llama.cpp翻译] 上下文大小: {context_size}")
                
                # 检查文件是否存在
                if not os.path.exists(self.llama_cli_path):
                    timestamp_print(f"[llama.cpp翻译] 错误: llama-completion.exe 不存在: {self.llama_cli_path}")
                    return False
                
                if not os.path.exists(self.model_path):
                    timestamp_print(f"[llama.cpp翻译] 错误: 模型文件不存在: {self.model_path}")
                    return False
                
                # 启动进程
                timestamp_print("[llama.cpp翻译] 正在启动进程...")
                self.process = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding='utf-8',
                    errors='ignore',
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
                )

                self.stdin = self.process.stdin
                self.stdout = self.process.stdout
                self.stderr = self.process.stderr

                timestamp_print("[llama.cpp翻译] 进程已启动，进程ID: " + str(self.process.pid))
                
                # 等待进程启动
                timestamp_print("[llama.cpp翻译] 等待进程初始化...")
                time.sleep(5)  # 增加等待时间确保进程完全启动
                
                # 检查进程是否仍在运行
                if self.process and self.process.poll() is None:
                    timestamp_print("[llama.cpp翻译] 服务器进程启动成功")
                    
                    # 尝试读取初始输出
                    try:
                        if self.stderr:
                            # 尝试读取错误输出
                            import msvcrt
                            if msvcrt.kbhit():
                                stderr_output = self.stderr.read(1024)
                                if stderr_output:
                                    timestamp_print(f"[llama.cpp翻译] 服务器启动错误输出: {stderr_output[:200]}...")
                    except Exception as e:
                        pass
                    
                    # 不发送测试命令，直接返回成功
                    # 测试命令可能会导致卡住，我们将在实际翻译时处理错误
                    timestamp_print("[llama.cpp翻译] 服务器进程启动成功，跳过测试命令")
                    return True
                else:
                    exit_code = self.process.poll() if self.process else "None"
                    timestamp_print(f"[llama.cpp翻译] 服务器进程启动后立即退出，退出码: {exit_code}")
                    
                    # 尝试读取错误输出
                    try:
                        if self.stderr:
                            stderr_output = self.stderr.read(1024)
                            if stderr_output:
                                timestamp_print(f"[llama.cpp翻译] 退出原因: {stderr_output}")
                    except Exception as e:
                        pass
                    
                    self.stop_process()
                    
            except Exception as e:
                timestamp_print(f"[llama.cpp翻译] 启动服务器进程失败: {str(e)}")
                import traceback
                traceback.print_exc()
                self.stop_process()
            
            # 等待一段时间后重试
            if attempt < max_attempts:
                timestamp_print(f"[llama.cpp翻译] 5秒后重试...")
                time.sleep(5)
        
        timestamp_print("[llama.cpp翻译] 服务器进程启动失败，达到最大尝试次数")
        return False

    def _read_output(self, timeout=60.0):
        """读取进程输出 - 持续尝试直到成功，添加超时机制"""
        if not self.stdout:
            timestamp_print("[llama.cpp翻译] 错误: 标准输出流未初始化")
            return b""

        output = []
        start_time = time.time()
        
        # 确保进程仍在运行
        if not self.process or self.process.poll() is not None:
            exit_code = self.process.poll() if self.process else "None"
            timestamp_print(f"[llama.cpp翻译] 错误: 服务器进程已退出，退出码: {exit_code}")
            # 尝试读取错误输出
            try:
                if self.stderr:
                    stderr_output = self.stderr.read(1024)
                    if stderr_output:
                        timestamp_print(f"[llama.cpp翻译] 进程退出原因: {stderr_output}")
            except Exception as e:
                pass
            return b""
        
        timestamp_print(f"[llama.cpp翻译] 开始读取服务器输出 (超时: {timeout}秒)...")
        
        # 循环读取直到超时或收到足够的输出
        while time.time() - start_time < timeout:
            # 检查进程是否仍在运行
            if self.process and self.process.poll() is not None:
                exit_code = self.process.poll()
                timestamp_print(f"[llama.cpp翻译] 错误: 服务器进程在读取过程中退出，退出码: {exit_code}")
                # 尝试读取错误输出
                try:
                    if self.stderr:
                        stderr_output = self.stderr.read(1024)
                        if stderr_output:
                            timestamp_print(f"[llama.cpp翻译] 进程退出原因: {stderr_output}")
                except Exception as e:
                    pass
                break
            
            try:
                # 尝试读取输出
                if os.name == 'nt':
                    # Windows 平台：使用更简单的读取方式
                    # 使用非阻塞读取
                    import msvcrt
                    
                    # 检查是否有数据可读
                    if msvcrt.kbhit():
                        # 尝试读取数据
                        data = self.stdout.read(1024)
                        if data:
                            output.append(data.encode('utf-8', errors='ignore'))
                            timestamp_print(f"[llama.cpp翻译] 读取到 {len(data)} 字节数据")
                            # 检查是否已收到足够的输出
                            output_str = b''.join(output).decode('utf-8', errors='ignore')
                            if len(output_str) > 50 or '\n' in output_str:
                                timestamp_print("[llama.cpp翻译] 收到完整输出")
                                break
                        else:
                            timestamp_print("[llama.cpp翻译] 警告: 读取到空数据")
                    else:
                        # 没有数据可读，短暂睡眠
                        time.sleep(0.1)
                else:
                    # Unix 平台：使用 select
                    import select
                    ready, _, _ = select.select([self.stdout], [], [], 0.1)
                    if ready:
                        data = self.stdout.read(1024)
                        if data:
                            output.append(data.encode('utf-8', errors='ignore'))
                            timestamp_print(f"[llama.cpp翻译] 读取到 {len(data)} 字节数据")
                            # 检查是否已收到足够的输出
                            output_str = b''.join(output).decode('utf-8', errors='ignore')
                            if len(output_str) > 50 or '\n' in output_str:
                                timestamp_print("[llama.cpp翻译] 收到完整输出")
                                break
            except Exception as e:
                timestamp_print(f"[llama.cpp翻译] 读取输出时发生异常: {str(e)}")
                # 不打印完整堆栈，避免输出过多
                # 继续尝试读取
                time.sleep(0.1)
            
            # 检查超时
            elapsed = time.time() - start_time
            remaining = timeout - elapsed
            if remaining < 0:
                break
        
        result = b''.join(output)
        if result:
            output_str = result.decode('utf-8', errors='ignore')
            timestamp_print(f"[llama.cpp翻译] 收到输出: {output_str[:100]}...")
        else:
            timestamp_print("[llama.cpp翻译] 错误: 未收到任何输出")
            # 尝试读取错误输出
            try:
                if self.stderr:
                    stderr_output = self.stderr.read(1024)
                    if stderr_output:
                        timestamp_print(f"[llama.cpp翻译] 错误输出: {stderr_output}")
            except Exception as e:
                pass
        
        return result

    def _send_command(self, command):
        """发送命令到服务器 - 使用标准输入"""
        if not self.stdin or not self.process or self.process.poll() is not None:
            timestamp_print("[llama.cpp翻译] 服务器进程未就绪，无法发送命令")
            # 尝试重新启动进程
            if not self.start_process():
                return False

        try:
            timestamp_print("[llama.cpp翻译] 发送命令到服务器...")
            # 发送命令
            self.stdin.write(command + '\n')
            self.stdin.flush()
            timestamp_print("[llama.cpp翻译] 命令发送成功")
            return True
        except Exception as e:
            timestamp_print(f"[llama.cpp翻译] 发送命令失败: {str(e)}")
            # 尝试重新启动进程
            if self.start_process():
                # 重新发送命令
                try:
                    self.stdin.write(command + '\n')
                    self.stdin.flush()
                    timestamp_print("[llama.cpp翻译] 重新发送命令成功")
                    return True
                except Exception as e2:
                    timestamp_print(f"[llama.cpp翻译] 重新发送命令失败: {str(e2)}")
                    return False
            return False

    def stop_process(self):
        """停止持久化进程"""
        if self.process and self.process.poll() is None:
            try:
                # 发送退出命令
                if self.stdin:
                    try:
                        self.stdin.write('\\q\n')
                        self.stdin.flush()
                    except:
                        pass
                
                # 等待进程退出
                self.process.wait(timeout=5)
            except Exception as e:
                timestamp_print(f"[llama.cpp翻译] 停止进程失败: {str(e)}")
                # 强制终止
                try:
                    self.process.terminate()
                    self.process.wait(timeout=3)
                except:
                    pass
            finally:
                self.process = None
                self.stdin = None
                self.stdout = None
                self.stderr = None
                timestamp_print("[llama.cpp翻译] 服务器进程已停止")

    def _find_llama_cli(self):
        """查找 llama-cli.exe 或 llama-completion.exe 路径"""
        # 优先使用 llama-completion（非交互式，更稳定）
        possible_paths = [
            os.path.join(PROJECT_ROOT, "llama_cpp", "llama-completion.exe"),
            os.path.join(PROJECT_ROOT, "llama_cpp", "llama-cli.exe"),
            os.path.join(PROJECT_ROOT, "llama_cpp", "build", "llama-completion.exe"),
            os.path.join(PROJECT_ROOT, "llama_cpp", "build", "llama-cli.exe"),
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path

        # 搜索 llama_cpp 目录下的所有可执行文件
        for exe_name in ["llama-completion.exe", "llama-cli.exe"]:
            search_pattern = os.path.join(PROJECT_ROOT, "llama_cpp", "**", exe_name)
            found = glob.glob(search_pattern, recursive=True)
            if found:
                return found[0]

        return None

    def _find_gguf_model(self):
        """查找 GGUF 模型文件 - 根据配置选择量化版本"""
        from config import config
        
        # 获取量化配置
        quantization = config.get('translator_quantization', 'auto')
        timestamp_print(f"[llama.cpp翻译] 量化配置: {quantization}")
        
        # 优先查找腾讯翻译模型
        model_dirs = [
            os.path.join(MODEL_CACHE_DIR, "tencent--HY-MT1.5-7B-GGUF"),
            os.path.join(MODEL_CACHE_DIR, "HY-MT1.5-7B-GGUF"),
        ]

        for model_dir in model_dirs:
            if os.path.exists(model_dir):
                # 查找 .gguf 文件
                gguf_files = glob.glob(os.path.join(model_dir, "*.gguf"))
                if gguf_files:
                    # 如果指定了具体量化版本，优先选择该版本
                    if quantization != "auto":
                        for f in gguf_files:
                            if quantization in f:
                                timestamp_print(f"[llama.cpp翻译] 选择指定量化模型: {os.path.basename(f)}")
                                return f
                        # 如果没有找到指定版本，打印警告并回退到自动选择
                        timestamp_print(f"[llama.cpp翻译警告] 未找到指定量化版本 {quantization}，回退到自动选择")
                    
                    # 自动选择：优先选择量化程度最高的版本（体积最小）
                    for preferred in ["Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L", "Q4_0", "Q4_K_S", "Q4_K_M", "Q5_0", "Q5_K_S", "Q5_K_M", "Q6_K", "Q8_0"]:
                        for f in gguf_files:
                            if preferred in f:
                                timestamp_print(f"[llama.cpp翻译] 自动选择模型: {os.path.basename(f)}")
                                return f
                    # 如果没有找到首选版本，返回第一个
                    timestamp_print(f"[llama.cpp翻译] 使用默认模型: {os.path.basename(gguf_files[0])}")
                    return gguf_files[0]

        # 搜索所有 GGUF 文件
        search_pattern = os.path.join(MODEL_CACHE_DIR, "**", "*.gguf")
        found = glob.glob(search_pattern, recursive=True)
        if found:
            # 如果指定了具体量化版本，尝试匹配
            if quantization != "auto":
                for f in found:
                    if quantization in f:
                        timestamp_print(f"[llama.cpp翻译] 选择指定量化模型: {os.path.basename(f)}")
                        return f
            # 选择最小的文件
            smallest_file = min(found, key=os.path.getsize)
            timestamp_print(f"[llama.cpp翻译] 自动选择最小模型: {os.path.basename(smallest_file)}")
            return smallest_file

        return None

    def translate(self, text, source_lang="en", target_lang="zh", context_size=20000):
        """翻译单个文本 - 使用与批量翻译相同的exe"""
        # 直接调用批量翻译的回退方法，确保使用相同的exe
        return self._translate_multi_fallback([text], source_lang, target_lang, context_size)[0]

    def _find_repeated_sequences(self, text: str, min_repeat: int = 2) -> list:
        """
        自动识别文本中的重复字符序列
        
        返回: [(重复序列, 重复次数, 起始位置), ...]
        """
        sequences = []
        i = 0
        n = len(text)
        
        while i < n:
            # 处理单字符重复（包括标点符号）
            if n - i >= 3:
                # 检查单字符重复
                current_char = text[i]
                repeat_count = 1
                j = i + 1
                while j < n and text[j] == current_char:
                    repeat_count += 1
                    j += 1
                if repeat_count >= 3:  # 单字符重复至少3次（包括标点符号）
                    sequences.append((current_char, repeat_count, i))
                    i = j
                    continue
            
            # 特殊处理包含标点的单字符重复序列
            # 检查常见的单字符+标点模式
            if n - i >= 2:
                # 检查模式: 字符+标点
                if i + 4 <= n:
                    # 检查 あ、あ、 模式
                    if text[i:i+2] == text[i+2:i+4]:
                        seq = text[i:i+2]  # 单字符+标点
                        repeat_count = 2
                        j = i + 4
                        while j + 2 <= n and text[j:j+2] == seq:
                            repeat_count += 1
                            j += 2
                        if repeat_count >= 3:  # 重复次数达到3次及以上才进行压缩
                            sequences.append((seq, repeat_count + 1, i))
                            i = j
                            continue
            
            # 双字符重复和双字符+标点重复已合并到长序列重复处理中
            
            # 尝试不同长度的序列
            # 从较长的序列开始检查，这样可以识别包含标点的重复序列
            found = False
            for seq_len in range(min(10, n - i), 1, -1):  # 从长到短检查，包括长度为2的序列
                seq = text[i:i + seq_len]
                
                # 计算该序列连续重复的次数
                repeat_count = 1
                j = i + seq_len
                while j + seq_len <= n and text[j:j + seq_len] == seq:
                    repeat_count += 1
                    j += seq_len
                
                # 如果重复次数达到阈值，记录该序列
                if repeat_count >= 3:
                    sequences.append((seq, repeat_count, i))
                    i = j  # 跳过已处理的重复部分
                    found = True
                    break
            
            # 如果没有找到重复序列，移动到下一个字符
            if not found:
                i += 1
        
        return sequences

    def _compress_repeated_sequences(self, text: str, keep_count: int = 2) -> str:
        """
        压缩文本中的重复字符序列
        
        Args:
            text: 原始文本
            keep_count: 保留的重复次数（默认保留2个）
        
        Returns:
            压缩后的文本
        """
        if not text:
            return text
        
        sequences = self._find_repeated_sequences(text)
        
        if not sequences:
            return text
        
        # 按位置排序（从后往前处理，避免位置偏移问题）
        sequences.sort(key=lambda x: x[2], reverse=True)
        
        result = text
        for seq, repeat_count, start_pos in sequences:
            # 只保留指定数量的重复
            if len(seq) == 3 and seq[-1] in ',!?.':
                # 对于双字符+标点的序列（长度为3），特殊处理
                # 保留 keep_count 个序列，但去掉最后一个标点
                keep_seq = (seq * keep_count)[:-1]
            elif len(seq) == 2 and seq[-1] in ',!?.':
                # 对于单字符+标点的序列（长度为2），特殊处理
                # 保留 keep_count 个序列，但去掉最后一个标点
                keep_seq = (seq * keep_count)[:-1]
            elif len(seq) == 1:
                # 对于单字符重复（包括标点符号），直接保留指定数量
                keep_seq = seq * keep_count
            else:
                # 其他序列正常处理
                keep_seq = seq * keep_count
            
            # 计算实际的原始长度，确保不超出文本长度
            original_len = len(seq) * repeat_count
            # 确保原始长度不超出文本长度
            max_len = len(result) - start_pos
            original_len = min(original_len, max_len)
            
            # 替换重复序列
            result = result[:start_pos] + keep_seq + result[start_pos + original_len:]
        
        # 清理可能的重复标点或字符
        # 例如：好的好的的 -> 好的好的
        cleaned = []
        i = 0
        n = len(result)
        while i < n:
            # 检查是否有连续重复的字符
            if i + 2 < n and result[i] == result[i+1] == result[i+2]:
                # 连续3个相同字符，只保留2个
                cleaned.append(result[i])
                cleaned.append(result[i+1])
                i += 3
            else:
                cleaned.append(result[i])
                i += 1
        
        return ''.join(cleaned)

    def preprocess_text(self, text, keep_count=2):
        """预处理文本，压缩重复字符序列
        
        对于包含重复字符的文本（如"ああああ..."、"うんうんうん..."），进行压缩处理
        将重复序列缩减为仅保留指定数量（默认2个）
        避免占用过多 token 导致其他文本无法翻译
        
        Args:
            text: 原始文本
            keep_count: 保留的重复次数（默认保留2个）
        
        Returns:
            压缩后的文本
        """
        if not text:
            return text
        
        # 使用重复字符压缩功能
        return self._compress_repeated_sequences(text, keep_count)

    def translate_multi(self, texts, source_lang="en", target_lang="zh", context_size=20000):
        """批量翻译多个文本 - 使用单次调用模式"""
        if not texts:
            return []
        
        # 直接使用单次调用模式，避免Windows上的进程通信问题
        return self._translate_multi_fallback(texts, source_lang, target_lang, context_size)

    def _translate_multi_fallback(self, texts, source_lang="en", target_lang="zh", context_size=20000):
        """使用单次调用模式进行翻译（回退方案）"""
        import tempfile
        
        # 预处理文本，截断超长重复内容
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # 构建翻译提示
        lang_map = {
            "zh": "Chinese",
            "en": "English",
            "ja": "Japanese",
            "ko": "Korean",
            "fr": "French",
            "de": "German",
            "es": "Spanish",
            "ru": "Russian",
            "ar": "Arabic",
            "hi": "Hindi",
            "pt": "Portuguese",
            "it": "Italian",
            "nl": "Dutch",
            "pl": "Polish"
        }

        source_lang_name = lang_map.get(source_lang, source_lang)
        target_lang_name = lang_map.get(target_lang, target_lang)

        if len(processed_texts) == 1:
            # 单条翻译提示词，与批量翻译保持一致的构建方式
            prompt_lines = [f"<|startoftext|>Translate this {source_lang_name} text to {target_lang_name}:"]
            prompt_lines.append(processed_texts[0])
            prompt_lines.append("")
            prompt_lines.append("Instructions:")
            prompt_lines.append("1. Translate the text completely")
            prompt_lines.append("2. Even if it's very short or just one character, translate it")
            prompt_lines.append("3. Include all punctuation and special characters")
            prompt_lines.append("4. Provide the FULL translation, not partial")
            prompt_lines.append("5. Maintain the original structure")
            prompt_lines.append("")
            prompt_lines.append("Translation:")
            prompt_lines.append("<|extra_0|>")
            prompt = "\n".join(prompt_lines)
        else:
            # 优化提示词，简洁明确
            num_texts = len(processed_texts)
            prompt_lines = [f"<|startoftext|>Translate these {num_texts} {source_lang_name} texts to {target_lang_name}."]
            prompt_lines.append("")
            prompt_lines.append("Instructions:")
            prompt_lines.append(f"1. Provide {num_texts} translations, one per input")
            prompt_lines.append(f"2. Number translations 1 to {num_texts}")
            prompt_lines.append("3. Do not skip any number")
            prompt_lines.append("4. Translate every text, no matter how short")
            prompt_lines.append("5. Keep the same order as input")
            prompt_lines.append("")
            prompt_lines.append("Input:")
            for i, text in enumerate(processed_texts, 1):
                prompt_lines.append(f"{i}. {text}")
            prompt_lines.append("")
            prompt_lines.append("Translations:")
            prompt = "\n".join(prompt_lines)

        # 打印输入模型的文本
        timestamp_print("[模型输入] 发送给翻译模型的完整输入:")
        print(prompt)
        print("[模型输入结束]")

        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
                prompt_file = f.name
                f.write(prompt)
            
            # 从配置获取翻译参数
            from config import config
            temperature = config.get('translation_temperature', 0.05)
            top_k = config.get('translation_top_k', 40)
            top_p = config.get('translation_top_p', 0.95)
            repetition_penalty = config.get('translation_repetition_penalty', 1.0)
            presence_penalty = config.get('translation_presence_penalty', 0.0)
            frequency_penalty = config.get('translation_frequency_penalty', 0.0)
            min_p = config.get('translation_min_p', 0.05)
            
            cmd = [
                self.llama_cli_path,
                "-m", self.model_path,
                "-f", prompt_file,
                "-n", "-1",
                "-t", "8",
                "-c", str(context_size),
                "--temp", str(temperature),
                "--top-p", str(top_p),
                "--top-k", str(top_k),
                "--min-p", str(min_p),
                "--repeat-penalty", str(repetition_penalty),
                "--presence-penalty", str(presence_penalty),
                "--frequency-penalty", str(frequency_penalty),
                "-ngl", "99",
                "--no-display-prompt",
                "--simple-io",
                "-no-cnv",
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='ignore',
                timeout=300
            )

            try:
                os.unlink(prompt_file)
            except:
                pass

            if result.returncode != 0:
                raise RuntimeError(f"llama-cli 运行失败: {result.stderr}")

            output = result.stdout.strip()
            
            # 打印模型输出的原始文本
            timestamp_print("[模型输出] 翻译模型的原始输出:")
            print(output)
            print("[模型输出结束]")
            
            special_tokens = [
                "<|startoftext|>", "<|endoftext|>", "<|eos|>", 
                "<|extra_0|>", "<|extra_4|>", "<|pad|>",
                " [end of text]", "[end of text]"
            ]
            for token in special_tokens:
                output = output.replace(token, "")
            
            if len(texts) == 1:
                translation = output.strip()
                translation = " ".join(translation.split())
                return [translation]
            else:
                # 解析带编号的翻译结果
                import re
                lines = [line.strip() for line in output.split('\n') if line.strip()]
                translations = []
                
                # 首先尝试按编号解析 (如 "1. 翻译内容" 或 "1) 翻译内容")
                numbered_trans = {}
                # 同时收集所有非编号行，用于回退解析
                non_numbered_lines = []
                
                for line in lines:
                    # 匹配编号格式：数字 + 标点 + 空格
                    match = re.match(r'^(\d+)[\.\)\:\]]\s*(.+)$', line)
                    if match:
                        id_num = int(match.group(1))
                        content = match.group(2).strip()
                        if content:
                            numbered_trans[id_num] = " ".join(content.split())
                    else:
                        # 收集非编号行
                        lower_line = line.lower()
                        if not any(skip in lower_line for skip in ["translate", "translations", "input"]):
                            cleaned = line.strip()
                            if cleaned:
                                non_numbered_lines.append(cleaned)
                
                # 按顺序提取翻译
                for i in range(1, len(texts) + 1):
                    if i in numbered_trans:
                        translations.append(numbered_trans[i])
                    else:
                        translations.append(None)
                
                # 如果编号解析缺失，尝试使用非编号行填充
                if None in translations and non_numbered_lines:
                    timestamp_print(f"[llama.cpp翻译诊断] 尝试使用非编号行填充缺失的翻译")
                    line_idx = 0
                    for i in range(len(translations)):
                        if translations[i] is None and line_idx < len(non_numbered_lines):
                            translations[i] = non_numbered_lines[line_idx]
                            line_idx += 1
                
                # 诊断信息：显示解析结果
                timestamp_print(f"[llama.cpp翻译诊断] 输入片段数: {len(texts)}, 解析到翻译数: {len([t for t in translations if t is not None])}")
                timestamp_print(f"[llama.cpp翻译诊断] 模型输出行数: {len(lines)}")
                
                # 如果编号解析缺失太多，尝试直接按行解析
                missing_count = translations.count(None)
                if missing_count > len(texts) * 0.3:  # 缺失超过30%
                    timestamp_print(f"[llama.cpp翻译] 编号解析缺失 {missing_count} 个，尝试直接按行解析")
                    timestamp_print(f"[llama.cpp翻译诊断] 前5行输出: {lines[:5]}")
                    translations = []
                    for line in lines:
                        lower_line = line.lower()
                        # 跳过提示词
                        if any(skip in lower_line for skip in ["translate", "translations"]):
                            continue
                        # 移除编号前缀
                        cleaned = re.sub(r'^\d+[\.\)\:\]]\s*', '', line).strip()
                        if cleaned:
                            translations.append(" ".join(cleaned.split()))
                    timestamp_print(f"[llama.cpp翻译诊断] 按行解析后得到 {len(translations)} 个翻译")
                
                # 填充缺失的翻译，并记录哪些使用了原文
                filled_count = 0
                for i in range(len(translations)):
                    if translations[i] is None:
                        timestamp_print(f"[llama.cpp翻译诊断] 片段 {i+1} 使用原文填充: {texts[i][:30]}...")
                        translations[i] = texts[i]
                        filled_count += 1
                
                if filled_count > 0:
                    timestamp_print(f"[llama.cpp翻译诊断] 共 {filled_count} 个片段使用原文填充")
                
                # 确保数量正确
                while len(translations) < len(texts):
                    idx = len(translations)
                    timestamp_print(f"[llama.cpp翻译诊断] 片段 {idx+1} 缺失，使用原文: {texts[idx][:30]}...")
                    translations.append(texts[idx])
                
                if len(translations) > len(texts):
                    translations = translations[:len(texts)]
                
                return translations

        except subprocess.TimeoutExpired:
            raise RuntimeError("llama-cli 翻译超时")
        except Exception as e:
            raise RuntimeError(f"llama-cli 翻译失败: {str(e)}")

    def estimate_tokens(self, text, language="en"):
        """估算文本的 token 数量
        
        根据不同语言使用不同的换算比例
        """
        if not text:
            return 0
        
        # 语言到换算比例的映射
        char_per_token = {
            "zh": 1.00,    # 中文
            "zh-Hant": 1.00, # 繁体中文
            "yue": 1.00,    # 粤语
            "ja": 1.5,     # 日语（调整为更准确的比例）
            "ko": 2.0,     # 韩语（调整为更准确的比例）
            "en": 4.23,    # 英文
            "fr": 3.69,    # 法语
            "pt": 4.36,    # 葡萄牙语
            "es": 4.45,    # 西班牙语
            "tr": 4.23,    # 土耳其语
            "ru": 3.36,    # 俄语
            "ar": 3.80,    # 阿拉伯语
            "th": 4.73,    # 泰语
            "it": 4.18,    # 意大利语
            "de": 4.80,    # 德语
            "vi": 4.64,    # 越南语
            "ms": 5.44,    # 马来语
            "id": 4.90,    # 印尼语
            "tl": 4.23,    # 菲律宾语
            "hi": 3.23,    # 印地语
            "pl": 4.46,    # 波兰语
            "cs": 4.73,    # 捷克语
            "nl": 4.80,    # 荷兰语
            "km": 5.44,    # 高棉语
            "my": 4.23,    # 缅甸语
            "fa": 3.80,    # 波斯语
            "gu": 4.73,    # 古吉拉特语
            "ur": 3.23,    # 乌尔都语
            "te": 3.23,    # 泰卢固语
            "mr": 4.73,    # 马拉地语
            "he": 3.23,    # 希伯来语
            "bn": 3.23,    # 孟加拉语
            "ta": 3.23,    # 泰米尔语
            "uk": 4.73,    # 乌克兰语
            "bo": 3.80,    # 藏语
            "kk": 3.23,    # 哈萨克语
            "mn": 3.23,    # 蒙古语
            "ug": 3.80     # 维吾尔语
        }
        
        # 获取对应语言的换算比例
        ratio = char_per_token.get(language, 4.0)  # 默认值
        
        # 计算token数
        return int(len(text) / ratio) + 1
    
    def calculate_batch_tokens(self, texts, source_lang="en", target_lang="zh"):
        """计算批次所需的 token 数量
        
        包括：
        1. 提示词本身的 token
        2. 所有输入文本的 token
        3. 预期输出的 token（每个文本的翻译）
        4. 编号格式和换行的开销
        """
        if not texts:
            return 0
        
        # 计算预处理后的文本长度
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # 提示词开销计算（根据实际提示词长度估算）
        num_texts = len(texts)
        source_lang_name = {"zh": "Chinese", "en": "English", "ja": "Japanese", "ko": "Korean"}.get(source_lang, source_lang)
        target_lang_name = {"zh": "Chinese", "en": "English", "ja": "Japanese", "ko": "Korean"}.get(target_lang, target_lang)
        
        # 构建实际提示词以更准确估算
        prompt_lines = [f"Translate the following {num_texts} {source_lang_name} texts to {target_lang_name}:"]
        prompt_lines.append("")
        for i, text in enumerate(processed_texts, 1):
            prompt_lines.append(f"{i}. {text}")
        prompt_lines.append("")
        prompt_lines.append(f"CRITICAL INSTRUCTION: You must provide exactly {num_texts} translations, one for each input text, in the same order.")
        prompt_lines.append("YOU MUST translate EVERY text, including short single characters and phrases.")
        prompt_lines.append("YOU MUST number each translation to match the input numbering exactly.")
        prompt_lines.append("YOU MUST NOT skip any translations, even for short or single-character inputs.")
        prompt_lines.append("YOU MUST continue until ALL translations are provided, including the last one.")
        prompt_lines.append("FAILURE TO PROVIDE ALL TRANSLATIONS WILL RESULT IN AN INCOMPLETE RESPONSE.")
        
        # 计算提示词token数
        prompt_text = "\n".join(prompt_lines)
        prompt_tokens = self.estimate_tokens(prompt_text, source_lang)
        
        # 输入文本的 tokens
        input_tokens = sum(self.estimate_tokens(text, source_lang) for text in processed_texts)
        
        # 预期输出的 tokens（翻译通常和原文长度相近或略短）
        output_tokens = sum(self.estimate_tokens(text, target_lang) for text in processed_texts)
        
        # 格式开销（每个片段的编号和换行）
        format_overhead = len(texts) * 3  # 每个片段约3个token的格式开销
        
        total_tokens = prompt_tokens + input_tokens + output_tokens + format_overhead
        return total_tokens
    
    def split_into_smart_batches(self, segments, max_tokens_per_batch=6000, max_batch_size=32, max_output_tokens=8000, source_lang="en", target_lang="zh"):
        """智能分割批次，确保不超出 token 限制和最大输出限制
        
        Args:
            segments: 所有片段列表
            max_tokens_per_batch: 每批最大 token 数（留一些余量给上下文）
            max_batch_size: 每批最大片段数（防止批次过大）
            max_output_tokens: 最大输出 token 数（llama.cpp 限制）
            source_lang: 源语言代码
            target_lang: 目标语言代码
        
        Returns:
            批次列表，每个批次是一个片段列表
        """
        batches = []
        current_batch = []
        current_batch_tokens = 0
        
        # 安全余量（预留10%的token空间）
        safety_margin = 0.9
        adjusted_max_tokens = int(max_tokens_per_batch * safety_margin)
        adjusted_max_output = int(max_output_tokens * safety_margin)
        
        for segment in segments:
            text = segment["text"]
            text_tokens = self.estimate_tokens(text, source_lang)
            
            # 检查加入这个片段是否会超出限制
            # 预估加入后的总 token 数
            projected_tokens = self.calculate_batch_tokens(
                [s["text"] for s in current_batch] + [text],
                source_lang, target_lang
            )
            
            # 预估输出 token 数（翻译输出）
            projected_output_tokens = sum(
                self.estimate_tokens(self.preprocess_text(s["text"]), target_lang) 
                for s in current_batch + [segment]
            ) + len(current_batch + [segment]) * 10  # 加上编号和格式开销
            
            # 检查是否超出限制
            batch_full = False
            
            # 检查token限制
            if projected_tokens > adjusted_max_tokens:
                batch_full = True
            
            # 检查输出token限制
            if projected_output_tokens > adjusted_max_output:
                batch_full = True
            
            # 检查批次大小限制
            if len(current_batch) >= max_batch_size:
                batch_full = True
            
            # 如果当前批次已满，创建新批次
            if batch_full:
                if current_batch:  # 保存当前批次
                    batches.append(current_batch)
                    batch_texts = [s["text"] for s in current_batch]
                    total_tokens = self.calculate_batch_tokens(batch_texts, source_lang, target_lang)
                    input_tokens = sum(self.estimate_tokens(self.preprocess_text(text), source_lang) for text in batch_texts)
                    actual_output = sum(self.estimate_tokens(self.preprocess_text(s["text"]), target_lang) for s in current_batch)
                    timestamp_print(f"[智能分批] 创建批次 {len(batches)}: {len(current_batch)} 个片段, "
                                  f"总token数 {total_tokens} (输入 {input_tokens} + 输出 {actual_output} + 开销)")
                
                # 开始新批次
                current_batch = [segment]
                current_batch_tokens = text_tokens
            else:
                # 加入当前批次
                current_batch.append(segment)
                current_batch_tokens += text_tokens
        
        # 保存最后一个批次
        if current_batch:
            batches.append(current_batch)
            batch_texts = [s["text"] for s in current_batch]
            total_tokens = self.calculate_batch_tokens(batch_texts, source_lang, target_lang)
            input_tokens = sum(self.estimate_tokens(self.preprocess_text(text), source_lang) for text in batch_texts)
            actual_output = sum(self.estimate_tokens(self.preprocess_text(s["text"]), target_lang) for s in current_batch)
            timestamp_print(f"[智能分批] 创建批次 {len(batches)}: {len(current_batch)} 个片段, "
                          f"总token数 {total_tokens} (输入 {input_tokens} + 输出 {actual_output} + 开销)")
        
        return batches

    def translate_batch(self, segments, source_lang="en", target_lang="zh", progress_callback=None, batch_size=None, context_size=2048, max_output_tokens=8000):
        """批量翻译 - 使用智能分批确保不超出上下文限制和输出限制"""
        from config import config
        
        # 从配置获取token限制（如果未指定）
        if batch_size is None:
            batch_size = config.get('translation_batch_size', 4096)
        
        total_segments_segments = len(segments)
        timestamp_print(f"[llama.cpp翻译] 开始批量翻译，共 {total_segments_segments} 个片段")
        timestamp_print(f"[llama.cpp翻译] 上下文大小: {context_size}, 最大输出: {max_output_tokens} tokens")
        timestamp_print(f"[llama.cpp翻译] 每批最大token数: {batch_size}")
        
        # 使用智能分批算法
        # 使用配置的token限制作为每批的最大token数
        max_tokens_per_batch = batch_size
        # 最大批次大小限制
        max_batch_size = 32
        batches = self.split_into_smart_batches(segments, max_tokens_per_batch, max_batch_size, max_output_tokens, source_lang, target_lang)
        
        timestamp_print(f"[llama.cpp翻译] 智能分批完成，共 {len(batches)} 个批次")
        
        # 处理每个批次
        processed_count = 0
        for batch_num, batch in enumerate(batches, 1):
            timestamp_print(f"[llama.cpp翻译] 处理第 {batch_num}/{len(batches)} 批，包含 {len(batch)} 个片段")
            
            # 提取该批次的所有原文
            texts = [segment["text"] for segment in batch]
            
            # 直接调用 _translate_batch_with_retry 方法，处理未翻译的片段
            try:
                translations = self._translate_batch_with_retry(
                    batch, texts, source_lang, target_lang, context_size, max_output_tokens
                )
                
                # 将翻译结果赋值给对应的片段
                for i, (segment, translated) in enumerate(zip(batch, translations)):
                    segment["translated"] = translated
            except Exception as e:
                timestamp_print(f"[llama.cpp翻译] 翻译失败，尝试逐条翻译: {str(e)}")
                # 批量失败时，逐条翻译
                for i, segment in enumerate(batch):
                    original_text = segment["text"]
                    try:
                        translated_text = self.translate(original_text, source_lang, target_lang, context_size)
                        segment["translated"] = translated_text
                    except Exception as e2:
                        timestamp_print(f"[llama.cpp翻译] 翻译失败: {str(e2)}")
                        segment["translated"] = original_text
            
            # 更新处理计数
            for segment in batch:
                processed_count += 1
                if progress_callback and total_segments_segments > 0:
                    progress_callback(int(processed_count / total_segments_segments * 100))
            
            # 计算实际tokens（使用与预估值相同的预处理）
            actual_input_tokens = sum(self.estimate_tokens(self.preprocess_text(segment["text"])) for segment in batch)
            actual_output_tokens = sum(self.estimate_tokens(self.preprocess_text(segment.get("translated", segment["text"]))) for segment in batch)
            
            # 批量打印这一批的翻译结果
            batch_start_idx = processed_count - len(batch) + 1
            batch_end_idx = processed_count
            timestamp_print(f"[翻译批次 {batch_num}/{len(batches)}] 完成 {batch_start_idx}-{batch_end_idx}/{total_segments_segments}:")
            timestamp_print(f"  [Tokens] 实际输入: {actual_input_tokens}, 实际输出: {actual_output_tokens}")
            for i, segment in enumerate(batch):
                global_idx = batch_start_idx + i
                original = segment["text"]
                translated = segment.get("translated") or original
                # 截断过长的文本以便显示
                orig_display = original[:50] + "..." if len(original) > 50 else original
                trans_display = translated[:50] + "..." if len(translated) > 50 else translated
                print(f"  [{global_idx}/{total_segments_segments}] {orig_display} -> {trans_display}")
        
        timestamp_print(f"[llama.cpp翻译] 批量翻译完成，共翻译 {total_segments_segments} 个片段")
        return segments

    def _calculate_similarity(self, text1, text2):
        """计算两个文本的相似度（简单实现）"""
        if not text1 or not text2:
            return 0.0
        
        # 预处理：去除空格和标点，转为小写
        import re
        def normalize(text):
            text = re.sub(r'\s+', '', text.lower())
            text = re.sub(r'[^\w\u4e00-\u9fff]', '', text)  # 保留字母数字和中文
            return text
        
        norm1 = normalize(text1)
        norm2 = normalize(text2)
        
        if not norm1 or not norm2:
            return 0.0
        
        # 计算编辑距离相似度
        from difflib import SequenceMatcher
        similarity = SequenceMatcher(None, norm1, norm2).ratio()
        return similarity

    def _translate_batch_with_retry(self, batch, texts, source_lang, target_lang, context_size, max_output_tokens, retry_count=0):
        """翻译批次，对未翻译片段进行单条翻译"""
        max_retries = 3
        
        try:
            # 使用批量翻译
            translations = self.translate_multi(texts, source_lang, target_lang, context_size)
            
            # 检测翻译质量，记录未翻译的片段索引
            untranslated_indices = []
            for i, (orig, trans) in enumerate(zip(texts, translations)):
                similarity = self._calculate_similarity(orig, trans)
                # 根据文本长度调整相似度阈值
                text_length = len(orig)
                if text_length <= 2:
                    # 单个字符或两个字符，即使相似度高也视为已翻译
                    # 因为很多日文中的单字在中文中是相同的
                    threshold = 1.01  # 设置一个大于1的值，确保不会被视为未翻译
                elif text_length <= 5:
                    # 短文本，相似度阈值设为0.7
                    threshold = 0.7
                else:
                    # 普通文本，相似度阈值设为0.8
                    threshold = 0.8
                timestamp_print(f"[llama.cpp翻译] 片段 {i+1} 相似度: {similarity}, 阈值: {threshold}, 原文: {orig[:20]}..., 译文: {trans[:20]}...")
                if similarity > threshold:
                    untranslated_indices.append(i)
            
            # 处理未翻译的片段（只对未翻译的片段进行单条翻译）
            timestamp_print(f"[llama.cpp翻译] 未翻译片段索引: {untranslated_indices}")
            if untranslated_indices:
                timestamp_print(f"[llama.cpp翻译] 检测到 {len(untranslated_indices)} 个未翻译片段，只对这些片段进行单条翻译")
                
                for idx in untranslated_indices:
                    text = texts[idx]
                    timestamp_print(f"[llama.cpp翻译] 单条翻译未翻译片段: {text[:30]}...")
                    
                    try:
                        # 直接使用逐条翻译，只翻译这个未翻译的片段
                        text_length = len(text)
                        timestamp_print(f"[llama.cpp翻译] 开始单条翻译: {text} (长度: {text_length})")
                        single_translation = self.translate(text, source_lang, target_lang, context_size)
                        timestamp_print(f"[llama.cpp翻译] 单条翻译结果: {single_translation}")
                        # 再次检查翻译质量
                        similarity = self._calculate_similarity(text, single_translation)
                        # 根据文本长度调整阈值
                        if text_length <= 2:
                            threshold = 0.5
                        elif text_length <= 5:
                            threshold = 0.7
                        else:
                            threshold = 0.8
                        timestamp_print(f"[llama.cpp翻译] 单条翻译相似度: {similarity}, 阈值: {threshold}")
                        if similarity <= threshold:
                            translations[idx] = single_translation
                            timestamp_print(f"[llama.cpp翻译] 单条翻译成功: {single_translation[:30]}...")
                        else:
                            timestamp_print(f"[llama.cpp翻译] 单条翻译仍未成功，使用原文")
                    except Exception as e:
                        timestamp_print(f"[llama.cpp翻译] 单条翻译失败: {str(e)}")
            else:
                timestamp_print("[llama.cpp翻译] 没有检测到未翻译的片段")
            
            # 再次检查是否还有未翻译的片段
            final_untranslated = []
            for i, (orig, trans) in enumerate(zip(texts, translations)):
                similarity = self._calculate_similarity(orig, trans)
                # 根据文本长度调整相似度阈值
                text_length = len(orig)
                if text_length <= 2:
                    # 单个字符或两个字符，即使相似度高也视为已翻译
                    # 因为很多日文中的单字在中文中是相同的
                    threshold = 1.01  # 设置一个大于1的值，确保不会被视为未翻译
                elif text_length <= 5:
                    # 短文本，相似度阈值设为0.7
                    threshold = 0.7
                else:
                    # 普通文本，相似度阈值设为0.8
                    threshold = 0.8
                if similarity > threshold:
                    final_untranslated.append(i)
            
            if final_untranslated:
                timestamp_print(f"[llama.cpp翻译] 仍有 {len(final_untranslated)} 个片段未翻译，使用原文")
            else:
                # 所有片段都已翻译
                timestamp_print(f"[llama.cpp翻译] 批次翻译完成，所有片段都已翻译")
            
            return translations
            
        except Exception as e:
            if retry_count < max_retries:
                retry_count += 1
                timestamp_print(f"[llama.cpp翻译] 批量翻译失败，第{retry_count}次重试: {str(e)}")
                # 直接使用逐条翻译
                translations = []
                for text in texts:
                    try:
                        translated = self.translate(text, source_lang, target_lang, context_size)
                        translations.append(translated)
                    except Exception as e2:
                        timestamp_print(f"[llama.cpp翻译] 逐条翻译失败: {str(e2)}")
                        translations.append(text)
                return translations
            else:
                # 最终失败，使用逐条翻译
                timestamp_print(f"[llama.cpp翻译] 批量翻译失败，使用逐条翻译: {str(e)}")
                translations = []
                for text in texts:
                    try:
                        translated = self.translate(text, source_lang, target_lang, context_size)
                        translations.append(translated)
                    except Exception as e2:
                        timestamp_print(f"[llama.cpp翻译] 逐条翻译失败: {str(e2)}")
                        translations.append(text)
                return translations


def get_local_model_path(model_path):
    """获取本地模型路径"""
    # 处理各种可能的模型路径格式
    model_name = model_path.replace("tencent/", "").replace("tencent--", "").split("/")[-1]

    # 可能的本地路径列表
    possible_paths = [
        # 直接目录名
        os.path.join(MODEL_CACHE_DIR, model_name),
        # HuggingFace缓存格式
        os.path.join(MODEL_CACHE_DIR, f"tencent--{model_name}"),
        # 子目录格式
        os.path.join(MODEL_CACHE_DIR, "tencent", model_name),
        # 处理 HY-MT1.5-7B-GGUF 的特殊情况
        os.path.join(MODEL_CACHE_DIR, "tencent--HY-MT1.5-7B-GGUF"),
        os.path.join(MODEL_CACHE_DIR, "HY-MT1.5-7B-GGUF"),
    ]

    # 检查所有可能的路径
    for path in possible_paths:
        if os.path.exists(path) and os.path.isdir(path):
            timestamp_print(f"[翻译] 找到本地模型: {path}")
            return path

    # 检查原始路径
    direct_path = os.path.join(MODEL_CACHE_DIR, model_path.replace("/", "--"))
    if os.path.exists(direct_path) and os.path.isdir(direct_path):
        timestamp_print(f"[翻译] 找到本地模型: {direct_path}")
        return direct_path

    return None


def translate_text(recognized_result, model_path, device_choice="auto", progress_callback=None,
                   beam_size=1, max_length=256, target_language="zh", batch_size=32, context_size=None, max_output_tokens=8000):
    """翻译识别结果"""
    if not target_language:
        target_language = "zh"

    # 获取本地模型路径
    local_model_path = get_local_model_path(model_path)
    if not local_model_path:
        error_msg = f"本地模型不存在: {model_path}，请确保模型已在models目录中"
        timestamp_print(f"[错误信息] {error_msg}")
        raise FileNotFoundError(error_msg)

    # 如果没有指定上下文大小，从配置中获取
    if context_size is None:
        from config import config
        context_size = config.get('translation_context_size', 20000)

    # 使用 llama-cli.exe 运行 GGUF 模型
    return translate_with_llama_cpp(recognized_result, progress_callback, target_language, batch_size, context_size, max_output_tokens)


def translate_with_llama_cpp(recognized_result, progress_callback, target_language, batch_size=None, context_size=None, max_output_tokens=8000):
    """使用 llama-cli.exe 运行 GGUF 模型进行翻译"""
    from config import config
    
    # 从配置获取token限制（如果未指定）
    if batch_size is None:
        batch_size = config.get('translation_batch_size', 4096)
    
    timestamp_print(f"[llama.cpp翻译] 正在加载 GGUF 模型...")

    # 使用单例模式获取翻译器实例
    translator = LlamaCppTranslator()

    segments = recognized_result["segments"]
    total_segments_segments = len(segments)

    # 获取源语言
    source_language = recognized_result.get("language", "en")
    
    # 如果没有指定上下文大小，从配置中获取
    if context_size is None:
        context_size = config.get('translation_context_size', 10000)
    
    timestamp_print(f"[llama.cpp翻译] 开始处理 {total_segments_segments} 个片段...")
    timestamp_print(f"[llama.cpp翻译] 源语言: {source_language}, 目标语言: {target_language}")
    timestamp_print(f"[llama.cpp翻译] 上下文大小: {context_size}")
    timestamp_print(f"[llama.cpp翻译] 每批最大token数: {batch_size}")
    
    # 使用单次调用模式，避免Windows上的进程通信问题
    timestamp_print("[llama.cpp翻译] 使用单次调用模式，批量翻译")

    # 使用批量翻译
    try:
        # 智能分批
        # 使用配置的token限制作为每批的最大token数
        max_tokens_per_batch = batch_size
        # 最大批次数量不设限制，由token数控制
        max_batch_count = 64
        batches = translator.split_into_smart_batches(
            segments, 
            max_tokens_per_batch=max_tokens_per_batch, 
            max_batch_size=max_batch_count, 
            max_output_tokens=max_output_tokens,
            source_lang=source_language,
            target_lang=target_language
        )
        
        timestamp_print(f"[llama.cpp翻译] 智能分批完成，共 {len(batches)} 个批次")
        
        processed_count = 0
        for batch_num, batch in enumerate(batches, 1):
            timestamp_print(f"[llama.cpp翻译] 处理第 {batch_num}/{len(batches)} 批，包含 {len(batch)} 个片段")
            
            # 提取该批次的所有原文
            texts = [segment["text"] for segment in batch]
            
            try:
                # 使用批量翻译（包含未翻译片段的单条翻译）
                translations = translator._translate_batch_with_retry(
                    batch, texts, source_language, target_language, context_size, max_output_tokens
                )
                
                # 将翻译结果赋值给对应的片段
                for i, (segment, translated) in enumerate(zip(batch, translations)):
                    segment["translated"] = translated
                    processed_count += 1
                    if progress_callback and total_segments_segments > 0:
                        progress_callback(int(processed_count / total_segments_segments * 100))
                
            except Exception as e:
                timestamp_print(f"[llama.cpp翻译] 批量翻译失败，回退到逐条翻译: {str(e)}")
                # 批量失败时，逐条翻译
                for i, segment in enumerate(batch):
                    original_text = segment["text"]
                    try:
                        translated_text = translator.translate(original_text, source_language, target_language, context_size)
                        segment["translated"] = translated_text
                    except Exception as e2:
                        timestamp_print(f"[llama.cpp翻译] 翻译失败: {str(e2)}")
                        segment["translated"] = original_text
                    
                    processed_count += 1
                    if progress_callback and total_segments_segments > 0:
                        progress_callback(int(processed_count / total_segments_segments * 100))
            
            # 计算实际tokens（使用与预估值相同的预处理）
            actual_input_tokens = sum(translator.estimate_tokens(translator.preprocess_text(segment["text"]), source_language) for segment in batch)
            actual_output_tokens = sum(translator.estimate_tokens(translator.preprocess_text(segment.get("translated", segment["text"])), target_language) for segment in batch)
            
            # 批量打印这一批的翻译结果
            batch_start_idx = processed_count - len(batch) + 1
            batch_end_idx = processed_count
            timestamp_print(f"[翻译批次 {batch_num}/{len(batches)}] 完成 {batch_start_idx}-{batch_end_idx}/{total_segments_segments}:")
            timestamp_print(f"  [Tokens] 实际输入: {actual_input_tokens}, 实际输出: {actual_output_tokens}")
            for i, segment in enumerate(batch):
                global_idx = batch_start_idx + i
                original = segment["text"]
                translated = segment.get("translated", original)
                # 截断过长的文本以便显示
                orig_display = original[:50] + "..." if len(original) > 50 else original
                trans_display = translated[:50] + "..." if len(translated) > 50 else translated
                print(f"  [{global_idx}/{total_segments_segments}] {orig_display} -> {trans_display}")
    
    except Exception as e:
        timestamp_print(f"[llama.cpp翻译] 翻译过程失败: {str(e)}")
        # 回退到逐条翻译
        for i, segment in enumerate(segments):
            original_text = segment["text"]
            try:
                translated_text = translator.translate(original_text, source_language, target_language, context_size)
                segment["translated"] = translated_text
            except Exception as e2:
                timestamp_print(f"[llama.cpp翻译] 翻译失败: {str(e2)}")
                segment["translated"] = original_text
            
            # 更新进度
            if progress_callback and total_segments_segments > 0:
                progress_callback(int((i + 1) / total_segments_segments * 100))
            
            # 打印翻译结果
            orig_display = original_text[:50] + "..." if len(original_text) > 50 else original_text
            trans_display = translated_text[:50] + "..." if len(translated_text) > 50 else translated_text
            print(f"  [{i+1}/{total_segments_segments}] {orig_display} -> {trans_display}")

    # 统计翻译结果
    translated_count = 0
    untranslated_count = 0
    untranslated_list = []
    
    for i, segment in enumerate(segments):
        original = segment.get('text', '')
        translated = segment.get('translated', '')
        
        # 检查是否翻译
        text_length = len(original)
        if text_length <= 2:
            # 单个字符或两个字符，即使与原文相同也视为已翻译
            # 因为很多日文中的单字在中文中是相同的
            if translated:
                translated_count += 1
            else:
                untranslated_count += 1
                untranslated_list.append((i + 1, original))
        else:
            # 普通文本，要求翻译结果与原文不同
            if translated and translated != original:
                translated_count += 1
            else:
                untranslated_count += 1
                untranslated_list.append((i + 1, original))
    
    # 打印详细的统计结果
    timestamp_print("\n翻译统计:")
    timestamp_print(f"总片段数: {total_segments_segments}")
    timestamp_print(f"已翻译: {translated_count}")
    timestamp_print(f"未翻译: {untranslated_count}")
    timestamp_print(f"翻译率: {translated_count/total_segments_segments*100:.1f}%")
    
    # 显示未翻译的片段
    if untranslated_list:
        timestamp_print(f"\n未翻译片段列表 (前20个):")
        for idx, text in untranslated_list[:20]:
            timestamp_print(f"  [{idx}] {text[:50]}...")
    
    # 翻译完成后保持进程运行，以便后续使用
    timestamp_print(f"[llama.cpp翻译] 翻译完成，共翻译 {total_segments_segments} 个片段")
    timestamp_print(f"[llama.cpp翻译] 服务器进程已保持，可用于后续翻译")

    # 返回完整的识别结果，包含语言信息
    return recognized_result
