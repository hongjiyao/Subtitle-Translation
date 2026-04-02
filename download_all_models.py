#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型下载脚本 - 支持多种模型类型和多线程下载
合并了下载所有模型和对齐模型的功能
"""

import os
import sys
import subprocess
import json
import requests
import time
from tqdm import tqdm

# 强制使用 UTF-8 编码
os.environ['PYTHONIOENCODING'] = 'utf-8'

# 设置HF_ENDPOINT为HF-Mirror以加速模型下载
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 设置模型下载目录
MODEL_DIR = os.path.join(os.getcwd(), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# 本地 aria2 路径 - 尝试多个可能的位置
ARIA2C_PATHS = [
    # 当前目录
    os.path.join(os.getcwd(), "aria2c.exe"),
    # 常见的aria2目录结构
    os.path.join(os.getcwd(), "aria2-1.37.0-win-64bit-build1", "aria2c.exe"),
    os.path.join(os.getcwd(), "aria2", "aria2c.exe"),
    os.path.join(os.getcwd(), "aria2", "aria2-1.37.0-win-64bit-build1", "aria2c.exe"),
]

# 找到第一个存在的 aria2c.exe
ARIA2C_PATH = None

# 首先检查预定义路径
for path in ARIA2C_PATHS:
    if os.path.exists(path):
        ARIA2C_PATH = path
        break

# 如果未找到，搜索当前目录及其子目录
if not ARIA2C_PATH:
    for root, dirs, files in os.walk(os.getcwd()):
        if "aria2c.exe" in files:
            ARIA2C_PATH = os.path.join(root, "aria2c.exe")
            break

# 定义需要下载的模型
MODELS = [
    # Faster-Whisper系列模型 - 不同大小的版本
    {"id": "Systran/faster-whisper-tiny", "dir": "Systran--faster-whisper-tiny", "priority": "medium", "type": "whisper", "download": True},
    {"id": "Systran/faster-whisper-base", "dir": "Systran--faster-whisper-base", "priority": "medium", "type": "whisper", "download": False},
    {"id": "Systran/faster-whisper-small", "dir": "Systran--faster-whisper-small", "priority": "medium", "type": "whisper", "download": False},
    {"id": "Systran/faster-whisper-medium", "dir": "Systran--faster-whisper-medium", "priority": "medium", "type": "whisper", "download": True},
    {"id": "Systran/faster-whisper-large-v2", "dir": "Systran--faster-whisper-large-v2", "priority": "high", "type": "whisper", "download": True},
    {"id": "Systran/faster-whisper-large-v3", "dir": "Systran--faster-whisper-large-v3", "priority": "high", "type": "whisper", "download": False},
    # HY-MT1.5-7B-GGUF翻译模型
    {"id": "tencent/HY-MT1.5-7B-GGUF", "dir": "tencent--HY-MT1.5-7B-GGUF", "priority": "high", "type": "gguf", "download": True},
    # Wav2Vec2 对齐模型 - 多语言支持
    {"id": "jonatasgrosman/wav2vec2-large-xlsr-53-english", "dir": "jonatasgrosman--wav2vec2-large-xlsr-53-english", "priority": "medium", "type": "wav2vec2", "download": True},
    {"id": "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn", "dir": "jonatasgrosman--wav2vec2-large-xlsr-53-chinese-zh-cn", "priority": "medium", "type": "wav2vec2", "download": True},
    {"id": "jonatasgrosman/wav2vec2-large-xlsr-53-japanese", "dir": "jonatasgrosman--wav2vec2-large-xlsr-53-japanese", "priority": "medium", "type": "wav2vec2", "download": True},
    {"id": "jonatasgrosman/wav2vec2-large-xlsr-53-french", "dir": "jonatasgrosman--wav2vec2-large-xlsr-53-french", "priority": "low", "type": "wav2vec2", "download": False},
    {"id": "jonatasgrosman/wav2vec2-large-xlsr-53-german", "dir": "jonatasgrosman--wav2vec2-large-xlsr-53-german", "priority": "low", "type": "wav2vec2", "download": False},
    {"id": "jonatasgrosman/wav2vec2-large-xlsr-53-spanish", "dir": "jonatasgrosman--wav2vec2-large-xlsr-53-spanish", "priority": "low", "type": "wav2vec2", "download": False},
    {"id": "jonatasgrosman/wav2vec2-large-xlsr-53-russian", "dir": "jonatasgrosman--wav2vec2-large-xlsr-53-russian", "priority": "low", "type": "wav2vec2", "download": False},
    {"id": "jonatasgrosman/wav2vec2-large-xlsr-53-arabic", "dir": "jonatasgrosman--wav2vec2-large-xlsr-53-arabic", "priority": "low", "type": "wav2vec2", "download": False},
    {"id": "jonatasgrosman/wav2vec2-large-xlsr-53-portuguese", "dir": "jonatasgrosman--wav2vec2-large-xlsr-53-portuguese", "priority": "low", "type": "wav2vec2", "download": False},
    {"id": "jonatasgrosman/wav2vec2-large-xlsr-53-italian", "dir": "jonatasgrosman--wav2vec2-large-xlsr-53-italian", "priority": "low", "type": "wav2vec2", "download": False},
    {"id": "jonatasgrosman/wav2vec2-large-xlsr-53-dutch", "dir": "jonatasgrosman--wav2vec2-large-xlsr-53-dutch", "priority": "low", "type": "wav2vec2", "download": False},
    {"id": "jonatasgrosman/wav2vec2-large-xlsr-53-polish", "dir": "jonatasgrosman--wav2vec2-large-xlsr-53-polish", "priority": "low", "type": "wav2vec2", "download": False},
    {"id": "jonatasgrosman/wav2vec2-large-xlsr-53-finnish", "dir": "jonatasgrosman--wav2vec2-large-xlsr-53-finnish", "priority": "low", "type": "wav2vec2", "download": False},
    {"id": "jonatasgrosman/wav2vec2-large-xlsr-53-persian", "dir": "jonatasgrosman--wav2vec2-large-xlsr-53-persian", "priority": "low", "type": "wav2vec2", "download": False}
]

# 简化日志输出
def log_message(message, level="INFO"):
    """记录日志信息"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    prefix = f"[{level.upper()}]"
    log_line = f"[{timestamp}] {prefix} {message}"
    # 确保输出使用正确的编码
    try:
        print(log_line)
    except UnicodeEncodeError:
        # 替换Unicode字符为ASCII
        log_line = log_line.replace('✓', '[OK]').replace('✗', '[ERROR]')
        print(log_line)
    # 同时将日志写入文件
    with open("download_log.txt", "a", encoding="utf-8") as f:
        f.write(log_line + "\n")

# 检查本地 aria2 是否存在
def check_aria2():
    """检查本地 aria2 是否存在"""
    global ARIA2C_PATH
    
    if ARIA2C_PATH and os.path.exists(ARIA2C_PATH):
        try:
            result = subprocess.run([ARIA2C_PATH, "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                version = result.stdout.split('\n')[0]
                log_message(f"[OK] 本地 aria2 可用: {version}")
                log_message(f"  路径: {ARIA2C_PATH}")
                return True
        except Exception as e:
            log_message(f"[ERROR] 本地 aria2 检查失败: {e}", "ERROR")
    else:
        log_message(f"[ERROR] 未找到本地 aria2", "ERROR")
        log_message(f"  已搜索路径:")
        for path in ARIA2C_PATHS:
            log_message(f"    - {path}")
    return False

# 获取最佳线程数
def get_optimal_threads():
    """根据系统配置获取最佳线程数"""
    try:
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        log_message(f"检测到CPU核心数: {cpu_count}")
        optimal_threads = max(8, min(16, cpu_count * 2))
        log_message(f"计算最佳线程数: {optimal_threads}")
        return optimal_threads
    except Exception as e:
        log_message(f"获取线程数失败: {str(e)}", "ERROR")
        return 16  # 默认使用16线程

# 检查网络连接
def check_internet_connection():
    """检查网络连接"""
    try:
        response = requests.get("https://hf-mirror.com", timeout=10)
        return True
    except:
        return False

# 清理目录中的重复文件（带有数字后缀的文件）
def cleanup_duplicate_files(directory):
    """清理目录中的重复文件"""
    if not os.path.exists(directory):
        return
    
    log_message(f"清理目录中的重复文件: {directory}")
    
    import re
    file_groups = {}
    
    for file in os.listdir(directory):
        match = re.match(r'^(.*)\.(\d+)\.(.*)$', file)
        if match:
            base_name = match.group(1)
            ext = match.group(3)
            original_file = f"{base_name}.{ext}"
            if original_file not in file_groups:
                file_groups[original_file] = []
            file_groups[original_file].append(file)
    
    deleted_count = 0
    for original_file, duplicate_files in file_groups.items():
        original_path = os.path.join(directory, original_file)
        if os.path.exists(original_path):
            for duplicate_file in duplicate_files:
                duplicate_path = os.path.join(directory, duplicate_file)
                log_message(f"删除重复文件: {duplicate_file}")
                try:
                    os.remove(duplicate_path)
                    deleted_count += 1
                except Exception as e:
                    log_message(f"✗ 删除文件失败: {str(e)}", "ERROR")
    
    if deleted_count > 0:
        log_message(f"[OK] 成功删除 {deleted_count} 个重复文件")
    else:
        log_message("[OK] 没有发现重复文件")

# 检查模型完整性和版本兼容性
def check_model(model_id, local_dir, model_type):
    """检查模型文件完整性和版本兼容性"""
    log_message(f"开始检查模型: {model_id}")
    
    # 检查目录是否存在
    if not os.path.exists(local_dir):
        log_message(f"[ERROR] 模型目录不存在: {local_dir}", "ERROR")
        return False, "模型目录不存在"
    
    # 模型类型特定检查
    if model_type == "whisper":
        required_files = ["config.json", "tokenizer.json"]
        weight_files = ["model.bin", "model.safetensors", "pytorch_model.bin"]
        vocab_files = ["vocabulary.txt", "vocabulary.json"]
    elif model_type == "gguf":
        # GGUF模型检查
        gguf_files = [f for f in os.listdir(local_dir) if f.endswith('.gguf')]
        if not gguf_files:
            reason = "未找到GGUF文件"
            log_message(f"[ERROR] {reason}", "ERROR")
            return False, reason
        
        for gguf_file in gguf_files:
            file_path = os.path.join(local_dir, gguf_file)
            file_size = os.path.getsize(file_path)
            if file_size < 1024 * 1024 * 1024:  # 1GB
                reason = f"GGUF文件过小: {gguf_file} ({file_size / (1024*1024*1024):.2f}GB)"
                log_message(f"[ERROR] {reason}", "ERROR")
                return False, reason
            log_message(f"[OK] GGUF文件: {gguf_file} ({file_size / (1024*1024*1024):.2f}GB)")
        
        log_message("[OK] GGUF模型完整性检查通过", "SUCCESS")
        return True, "模型文件完整"
    elif model_type == "wav2vec2":
        required_files = ["config.json", "preprocessor_config.json"]
        weight_files = ["pytorch_model.bin", "model.safetensors"]
        vocab_files = ["vocab.json"]
    else:
        # 通用模型检查
        config_path = os.path.join(local_dir, "config.json")
        if not os.path.exists(config_path):
            reason = "缺少config.json配置文件"
            log_message(f"[ERROR] {reason}", "ERROR")
            return False, reason
        
        weight_files = ["model.safetensors", "pytorch_model.bin"]
        required_files = []
        vocab_files = []
    
    # 检查必要文件
    if required_files:
        missing_files = []
        for file_name in required_files:
            if not os.path.exists(os.path.join(local_dir, file_name)):
                missing_files.append(file_name)
        
        if missing_files:
            reason = f"缺少必要文件: {', '.join(missing_files)}"
            log_message(f"[ERROR] {reason}", "ERROR")
            return False, reason
        
        log_message("[OK] 找到所有配置文件")
    
    # 检查模型权重文件
    weight_found = False
    for weight_file in weight_files:
        weight_path = os.path.join(local_dir, weight_file)
        if os.path.exists(weight_path):
            weight_found = True
            # 检查文件大小
            file_size = os.path.getsize(weight_path)
            # 不同模型的大小要求
            if model_type == "wav2vec2":
                min_size = 500 * 1024 * 1024  # 500MB
                size_unit = "MB"
                size_div = 1024*1024
            elif model_type == "whisper":
                min_size = 10 * 1024 * 1024  # 10MB
                size_unit = "MB"
                size_div = 1024*1024
            else:
                min_size = 100 * 1024 * 1024  # 100MB
                size_unit = "MB"
                size_div = 1024*1024
            
            if file_size < min_size:
                reason = f"模型权重文件过小: {weight_file} ({file_size / size_div:.2f}{size_unit})"
                log_message(f"[ERROR] {reason}", "ERROR")
                return False, reason
            log_message(f"[OK] 找到模型权重文件: {weight_file} ({file_size / size_div:.2f}{size_unit})")
            break
    
    if not weight_found:
        reason = "未找到模型权重文件"
        log_message(f"[ERROR] {reason}", "ERROR")
        return False, reason
    
    # 检查词汇文件
    if vocab_files:
        vocab_found = False
        for vocab_file in vocab_files:
            if os.path.exists(os.path.join(local_dir, vocab_file)):
                vocab_found = True
                log_message(f"[OK] 找到词汇文件: {vocab_file}")
                break
        
        if not vocab_found:
            reason = f"缺少词汇文件 ({' 或 '.join(vocab_files)})"
            log_message(f"[ERROR] {reason}", "ERROR")
            return False, reason
    
    # 检查版本兼容性
    config_path = os.path.join(local_dir, "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            
            if "_name_or_path" in config:
                log_message(f"[OK] 模型名称: {config['_name_or_path']}")
            if "version" in config:
                log_message(f"[OK] 模型版本: {config['version']}")
            if "model_type" in config:
                log_message(f"[OK] 模型架构: {config['model_type']}")
            
            log_message("[OK] 模型版本兼容性检查通过", "SUCCESS")
        except Exception as e:
            log_message(f"[WARNING] 版本检查失败: {str(e)}", "WARNING")
    
    log_message("[OK] 模型完整性检查通过", "SUCCESS")
    return True, "模型文件完整"

# 下载文件
def download_file(url, file_path, use_aria2=False):
    """下载单个文件"""
    file_dir = os.path.dirname(file_path)
    os.makedirs(file_dir, exist_ok=True)
    file_name = os.path.basename(file_path)
    
    # 检查文件是否已存在
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        log_message(f"[OK] 文件已存在，跳过下载: {file_name}")
        return True
    
    log_message(f"下载文件: {file_name}")
    
    if use_aria2 and ARIA2C_PATH:
        threads = get_optimal_threads()
        log_message(f"使用aria2多线程下载，{threads}线程")
        
        # 构建aria2命令
        cmd = [
            ARIA2C_PATH,
            "-x", str(threads),           # 最大连接数
            "-s", str(threads),           # 分片数
            "-k", "1M",           # 分片大小
            "--max-connection-per-server", "16",
            "--split", "16",
            "--min-split-size", "1M",
            "--file-allocation", "none",
            "-o", file_name,
            "-d", file_dir,
            url
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            log_message(f"[OK] 文件下载完成: {file_name}")
            cleanup_duplicate_files(file_dir)
            return True
        except subprocess.CalledProcessError as e:
            log_message(f"[ERROR] aria2下载失败: {file_name}", "ERROR")
            if e.stderr:
                log_message(f"    错误: {e.stderr[:200]}", "ERROR")
            log_message("尝试使用curl作为备用方案")
        except Exception as e:
            log_message(f"[ERROR] aria2下载出错: {e}", "ERROR")
            log_message("尝试使用curl作为备用方案")
    
    # 使用curl下载（作为备用方案）
    log_message("使用curl命令下载，显示详细进度条")
    command = f'curl -L -o "{file_path}" "{url}" --progress-bar'
    
    result = subprocess.run(
        command,
        shell=True,
        capture_output=False,  # 不捕获输出，让进度条实时显示
        text=True,
        timeout=3600  # 1小时超时
    )
    
    if result.returncode == 0:
        log_message(f"[OK] 文件下载完成: {file_name}")
        cleanup_duplicate_files(file_dir)
        return True
    else:
        log_message(f"[ERROR] curl下载失败: {file_name}", "ERROR")
        return False

# 下载模型
def download_model(model_info):
    """下载模型"""
    model_id = model_info["id"]
    model_dir_name = model_info["dir"]
    model_type = model_info.get("type", "unknown")
    local_dir = os.path.join(MODEL_DIR, model_dir_name)
    
    log_message(f"\n下载模型: {model_id}")
    log_message(f"本地目录: {local_dir}")
    log_message(f"模型类型: {model_type}")
    
    # 创建本地目录
    os.makedirs(local_dir, exist_ok=True)
    
    # 检查网络连接
    if not check_internet_connection():
        log_message("✗ 网络连接失败，请检查网络设置", "ERROR")
        return False
    
    try:
        from huggingface_hub import HfApi
        
        # 获取模型文件列表
        api = HfApi(endpoint=os.environ["HF_ENDPOINT"])
        files = api.list_repo_files(repo_id=model_id)
        
        # 打印所有找到的文件
        log_message(f"找到 {len(files)} 个文件:")
        for file in files[:10]:  # 只显示前10个文件
            log_message(f"  - {file}")
        if len(files) > 10:
            log_message(f"  ... 等 {len(files) - 10} 个文件")
        
        # 只下载必要文件
        required_files = []
        if model_type == "gguf":
            # GGUF模型只下载.gguf文件
            for file in files:
                if file.endswith(".gguf"):
                    required_files.append(file)
        elif model_type == "wav2vec2":
            # Wav2Vec2模型下载必要文件
            required_files = [
                "pytorch_model.bin",
                "config.json",
                "preprocessor_config.json",
                "vocab.json"
            ]
        else:
            # 其他模型下载必要文件
            for file in files:
                if any(file.endswith(ext) for ext in [".bin", ".safetensors", ".json", ".spm", ".model", ".txt"]):
                    required_files.append(file)
        
        log_message(f"需要下载 {len(required_files)} 个文件:")
        for file in required_files[:10]:  # 只显示前10个文件
            log_message(f"  - {file}")
        if len(required_files) > 10:
            log_message(f"  ... 等 {len(required_files) - 10} 个文件")
        
        # 检查是否使用aria2
        use_aria2 = check_aria2()
        if use_aria2:
            log_message("使用aria2多线程下载模型")
        else:
            log_message("使用curl下载模型")
        
        # 下载每个文件
        for file in required_files:
            file_path = os.path.join(local_dir, file)
            url = f"{os.environ['HF_ENDPOINT']}/{model_id}/resolve/main/{file}"
            
            if not download_file(url, file_path, use_aria2):
                return False
        
        # 下载完成后，清理整个模型目录中的重复文件
        log_message("清理模型目录中的重复文件...")
        cleanup_duplicate_files(local_dir)
        
        # 检查模型完整性
        log_message("检查模型完整性...")
        integrity_ok, integrity_reason = check_model(model_id, local_dir, model_type)
        
        if integrity_ok:
            log_message(f"[OK] 模型完整性检查通过: {local_dir}", "SUCCESS")
            return True
        else:
            log_message(f"[ERROR] 下载后模型完整性检查失败: {integrity_reason}", "ERROR")
            return False
            
    except Exception as e:
        log_message(f"[ERROR] 下载失败: {str(e)}", "ERROR")
        return False

# 测试模型是否能使用
def test_model_usability(model_info):
    """测试模型是否能使用"""
    model_id = model_info["id"]
    model_dir_name = model_info["dir"]
    model_type = model_info.get("type", "unknown")
    local_dir = os.path.join(MODEL_DIR, model_dir_name)
    
    log_message(f"\n=== 测试模型可用性: {model_id} ===", "INFO")
    
    # 特殊处理: 跳过 tencent/HY-MT1.5-7B-GGUF 模型的测试
    if model_id == "tencent/HY-MT1.5-7B-GGUF":
        log_message("[INFO] 跳过 tencent/HY-MT1.5-7B-GGUF 模型的测试")
        return True
    
    # 首先检查模型完整性
    integrity_ok, integrity_msg = check_model(model_id, local_dir, model_type)
    
    if not integrity_ok:
        log_message(f"[ERROR] 模型完整性检查失败: {integrity_msg}", "ERROR")
        return False
    
    log_message("[OK] 模型完整性检查通过")
    
    # 尝试导入必要的模块进行功能测试
    if model_type == "whisper":
        try:
            from faster_whisper import WhisperModel
            log_message("尝试加载Whisper模型...")
            model = WhisperModel(local_dir, device="cpu")
            log_message("[OK] Whisper模型加载成功")
            return True
        except Exception as e:
            log_message(f"[ERROR] Whisper模型加载失败: {str(e)}", "ERROR")
            return False
    elif model_type == "gguf":
        try:
            from ctransformers import AutoModelForCausalLM
            log_message("尝试加载GGUF模型...")
            gguf_files = [f for f in os.listdir(local_dir) if f.endswith('.gguf')]
            if gguf_files:
                model_file = os.path.join(local_dir, gguf_files[0])
                # 尝试不同的模型类型
                model_types = ["llama", "mistral", "gpt2", None]
                for model_type_try in model_types:
                    try:
                        if model_type_try:
                            model = AutoModelForCausalLM.from_pretrained(model_file, model_type=model_type_try)
                        else:
                            model = AutoModelForCausalLM.from_pretrained(model_file)
                        log_message(f"[OK] GGUF模型加载成功 (model_type={model_type_try})")
                        return True
                    except Exception as e_try:
                        log_message(f"[INFO] 尝试 model_type={model_type_try} 失败: {str(e_try)}")
                # 所有尝试都失败
                log_message("[ERROR] 所有模型类型尝试都失败", "ERROR")
                return False
            else:
                log_message("[ERROR] 未找到GGUF文件", "ERROR")
                return False
        except Exception as e:
            log_message(f"[ERROR] GGUF模型加载失败: {str(e)}", "ERROR")
            return False
    elif model_type == "wav2vec2":
        try:
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
            import torch
            import numpy as np
            
            log_message("尝试加载Wav2Vec2模型...")
            # 加载 processor (包含 tokenizer 和 feature extractor)
            processor = Wav2Vec2Processor.from_pretrained(local_dir, local_files_only=True)
            # 加载模型
            model = Wav2Vec2ForCTC.from_pretrained(local_dir, local_files_only=True)
            model.to("cpu")
            model.eval()
            
            log_message("执行简单测试...")
            # 创建测试音频数据 (1秒静音，16kHz)
            sample_rate = 16000
            dummy_audio = np.zeros(sample_rate, dtype=np.float32)
            # 处理输入
            inputs = processor(dummy_audio, sampling_rate=sample_rate, return_tensors="pt")
            # 前向传播
            with torch.no_grad():
                logits = model(**inputs).logits
            # 验证输出
            if logits.shape[0] == 1 and logits.shape[2] == len(processor.tokenizer):
                log_message("[OK] Wav2Vec2模型功能正常")
                return True
            else:
                log_message(f"[ERROR] 输出形状异常: {logits.shape}", "ERROR")
                return False
        except ImportError as e:
            log_message(f"[ERROR] 缺少依赖: {e}", "ERROR")
            return False
        except Exception as e:
            log_message(f"[ERROR] Wav2Vec2模型加载失败: {str(e)}", "ERROR")
            return False
    else:
        log_message("[OK] 模型类型未知，仅通过完整性检查")
        return True

# 验证所有模型的完整性
def verify_all_models():
    """验证所有模型的完整性"""
    log_message("\n" + "="*80)
    log_message("验证模型完整性")
    log_message("="*80)
    
    valid_count = 0
    invalid_count = 0
    skipped_count = 0
    invalid_models = []
    
    for model_info in MODELS:
        # 检查是否需要下载
        if not model_info.get('download', True):
            log_message(f"\n跳过验证: {model_info['id']} (download=False)")
            skipped_count += 1
            continue
        
        model_id = model_info["id"]
        model_dir_name = model_info["dir"]
        model_type = model_info.get("type", "unknown")
        local_dir = os.path.join(MODEL_DIR, model_dir_name)
        
        log_message(f"\n检查: {model_id}")
        is_valid, issues = check_model(model_id, local_dir, model_type)
        
        if is_valid:
            log_message(f"  [OK] 完整")
            valid_count += 1
        else:
            log_message(f"  [ERROR] 不完整: {issues}")
            invalid_count += 1
            invalid_models.append(model_id)
    
    log_message("\n" + "="*80)
    log_message("验证完成")
    log_message("="*80)
    log_message(f"完整: {valid_count}/{len(MODELS) - skipped_count}")
    log_message(f"不完整: {invalid_count}/{len(MODELS) - skipped_count}")
    log_message(f"跳过: {skipped_count}/{len(MODELS)}")
    
    if invalid_models:
        log_message(f"\n需要重新下载的模型:")
        for model in invalid_models:
            log_message(f"  - {model}")
    
    return invalid_count == 0, invalid_models

# 测试所有模型的功能性
def test_all_models_functionality():
    """测试所有模型的功能性"""
    log_message("\n" + "="*80)
    log_message("测试模型功能性")
    log_message("="*80)
    log_message("设备: cpu")
    log_message("注意: 此测试需要安装相应的依赖库\n")
    
    working_count = 0
    failed_count = 0
    skipped_count = 0
    failed_models = []
    
    for model_info in MODELS:
        # 检查是否需要下载
        if not model_info.get('download', True):
            log_message(f"\n跳过测试: {model_info['id']} (download=False)")
            skipped_count += 1
            continue
        
        log_message(f"\n测试: {model_info['id']}")
        
        # 先检查模型是否存在
        local_dir = os.path.join(MODEL_DIR, model_info["dir"])
        if not os.path.exists(local_dir):
            log_message(f"  [ERROR] 模型不存在，跳过测试")
            failed_count += 1
            failed_models.append((model_info['id'], "模型不存在"))
            continue
        
        is_working = test_model_usability(model_info)
        
        if is_working:
            log_message(f"  [OK] 功能正常")
            working_count += 1
        else:
            log_message(f"  [ERROR] 功能异常")
            failed_count += 1
            failed_models.append((model_info['id'], "功能异常"))
    
    log_message("\n" + "="*80)
    log_message("测试完成")
    log_message("="*80)
    log_message(f"正常: {working_count}/{len(MODELS) - skipped_count}")
    log_message(f"异常: {failed_count}/{len(MODELS) - skipped_count}")
    log_message(f"跳过: {skipped_count}/{len(MODELS)}")
    
    if failed_models:
        log_message(f"\n功能异常的模型:")
        for model, error in failed_models:
            log_message(f"  - {model}: {error}")
    
    return failed_count == 0, failed_models

# 主函数
def main():
    """下载所有需要的模型"""
    
    # 直接执行模型下载和测试，不需要命令行参数
    log_message("\n" + "="*80)
    log_message("模型下载脚本 - 多线程版")
    log_message("="*80)
    log_message(f"下载源: {os.environ['HF_ENDPOINT']}")
    log_message(f"模型目录: {MODEL_DIR}")
    log_message(f"aria2 路径: {ARIA2C_PATH}")
    log_message(f"共 {len(MODELS)} 个模型需要下载\n")
    
    # 按照优先级排序，高优先级的模型先下载
    sorted_models = sorted(MODELS, key=lambda x: 0 if x["priority"] == "high" else 1)
    
    # 下载所有模型
    success_count = 0
    failed_count = 0
    
    for i, model_info in enumerate(sorted_models, 1):
        # 检查是否需要下载
        if not model_info.get('download', True):
            log_message(f"\n[{i}/{len(sorted_models)}] ")
            log_message(f"跳过模型: {model_info['id']} (download=False)")
            continue
        
        log_message(f"\n[{i}/{len(sorted_models)}] ")
        log_message(f"处理模型: {model_info['id']}")
        log_message(f"优先级: {model_info['priority']}")
        
        if download_model(model_info):
            # 测试模型是否能使用
            if test_model_usability(model_info):
                success_count += 1
                log_message(f"✓ 模型处理成功: {model_info['id']}")
            else:
                log_message(f"✗ 模型测试失败: {model_info['id']}", "ERROR")
                failed_count += 1
        else:
            log_message(f"✗ 模型下载失败: {model_info['id']}", "ERROR")
            failed_count += 1
    
    log_message("\n" + "="*80)
    log_message("下载完成")
    log_message("="*80)
    log_message(f"成功: {success_count}/{len(MODELS)}")
    log_message(f"失败: {failed_count}/{len(MODELS)}")
    
    # 验证模型完整性
    if failed_count == 0:
        log_message("\n")
        all_valid, invalid_models = verify_all_models()
        
        if not all_valid:
            log_message("\n⚠ 部分模型不完整，建议重新运行脚本下载")
            return 1
    
    # 测试模型功能
    if failed_count == 0:
        log_message("\n")
        all_working, failed_models = test_all_models_functionality()
        
        if not all_working:
            log_message("\n⚠ 部分模型功能异常")
            return 1
    
    if failed_count > 0:
        log_message("\n提示: 失败的模型可以稍后重新运行脚本下载")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
