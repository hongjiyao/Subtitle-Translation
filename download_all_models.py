import os
import time
import subprocess
import requests
import json
import hashlib
from tqdm import tqdm

# 设置HF_ENDPOINT为HF-Mirror以加速模型下载
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 设置模型下载目录（直接放在当前目录下）
MODEL_DIR = os.path.join(os.getcwd(), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# 内存中存储文件信息的字典
file_info_cache = {}

# 初始化文件信息缓存
def initialize_file_info_cache():
    """初始化文件信息缓存，扫描所有已有的模型目录"""
    global file_info_cache
    log_message("初始化文件信息缓存...")
    
    if not os.path.exists(MODEL_DIR):
        return
    
    # 扫描所有模型目录
    for model_name in os.listdir(MODEL_DIR):
        model_dir = os.path.join(MODEL_DIR, model_name)
        if os.path.isdir(model_dir):
            # 更新并保存文件信息到缓存
            file_info = update_file_info(model_dir)
            file_info_cache[model_dir] = file_info

# 保存所有模型信息到根目录的.file_info.json
def save_all_file_info():
    """将所有模型的文件信息保存到项目根目录的.file_info.json"""
    global file_info_cache
    
    # 构建所有模型的信息字典
    all_models_info = {}
    
    for model_dir, file_info in file_info_cache.items():
        # 使用模型目录名作为键
        model_name = os.path.basename(model_dir)
        all_models_info[model_name] = {
            "directory": model_dir,
            "file_info": file_info,
            "timestamp": time.time()
        }
    
    # 保存到项目根目录
    root_file_info_path = os.path.join(os.getcwd(), ".file_info.json")
    try:
        with open(root_file_info_path, "w", encoding="utf-8") as f:
            json.dump(all_models_info, f, indent=2, ensure_ascii=False)
        log_message(f"✓ 所有模型信息已保存到: {root_file_info_path}")
    except Exception as e:
        log_message(f"✗ 保存文件信息失败: {str(e)}", "ERROR")

print(f"模型下载目录: {MODEL_DIR}")
print(f"使用HF-Mirror作为下载源: {os.environ['HF_ENDPOINT']}")
print("模型文件将直接下载到当前目录，不使用缓存")

# 简化日志输出，不再生成日志文件
def log_message(message, level="INFO"):
    """记录日志信息"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    
    # 根据级别设置前缀
    if level == "INFO":
        prefix = "[INFO]"
    elif level == "ERROR":
        prefix = "[ERROR]"
    elif level == "SUCCESS":
        prefix = "[SUCCESS]"
    else:
        prefix = f"[{level}]"
    
    log_entry = f"[{timestamp}] {prefix} {message}"
    print(log_entry)

def check_model_integrity(model_id, local_dir):
    """检查模型文件完整性"""
    log_message(f"开始检查模型完整性: {model_id}")
    log_message(f"检查目录: {local_dir}")
    
    # 检查目录是否存在
    if not os.path.exists(local_dir):
        log_message(f"✗ 模型目录不存在: {local_dir}", "ERROR")
        return False, "模型目录不存在"
    
    # 检查必要文件
    if "whisper" in model_id.lower():
        # Whisper模型检查
        return check_whisper_model_integrity(local_dir)
    elif "m2m100" in model_id.lower():
        # M2M-100翻译模型检查
        return check_m2m100_model_integrity(local_dir)
    else:
        # 通用模型检查
        return check_generic_model_integrity(local_dir)

def check_whisper_model_integrity(local_dir):
    """检查Whisper模型完整性"""
    # Faster-Whisper模型的必要文件
    required_files = [
        "config.json",
        "tokenizer.json"
    ]
    
    # 检查必要文件
    missing_files = []
    for file_name in required_files:
        file_path = os.path.join(local_dir, file_name)
        if not os.path.exists(file_path):
            missing_files.append(file_name)
    
    if missing_files:
        reason = f"缺少必要文件: {', '.join(missing_files)}"
        log_message(f"✗ {reason}", "ERROR")
        return False, reason
    
    # 检查模型权重文件
    weight_files = [
        "model.bin",  # Faster-Whisper使用model.bin
        "model.safetensors",
        "pytorch_model.bin"
    ]
    
    weight_found = False
    for weight_file in weight_files:
        weight_path = os.path.join(local_dir, weight_file)
        if os.path.exists(weight_path):
            weight_found = True
            # 检查文件大小
            file_size = os.path.getsize(weight_path)
            # Whisper模型至少应该有几十MB
            if file_size < 10 * 1024 * 1024:  # 10MB
                reason = f"模型权重文件过小: {weight_file} ({file_size / (1024*1024):.2f}MB)"
                log_message(f"✗ {reason}", "ERROR")
                return False, reason
            log_message(f"✓ 找到模型权重文件: {weight_file} ({file_size / (1024*1024):.2f}MB)")
            break
    
    if not weight_found:
        reason = "未找到模型权重文件 (model.bin、model.safetensors 或 pytorch_model.bin)"
        log_message(f"✗ {reason}", "ERROR")
        return False, reason
    
    # 检查词汇文件
    vocab_files = [
        "vocabulary.txt",
        "vocabulary.json"
    ]
    
    vocab_found = False
    for vocab_file in vocab_files:
        vocab_path = os.path.join(local_dir, vocab_file)
        if os.path.exists(vocab_path):
            vocab_found = True
            log_message(f"✓ 找到词汇文件: {vocab_file}")
            break
    
    if not vocab_found:
        reason = "缺少词汇文件 (vocabulary.txt 或 vocabulary.json)"
        log_message(f"✗ {reason}", "ERROR")
        return False, reason
    
    log_message("✓ Whisper模型完整性检查通过", "SUCCESS")
    return True, "模型文件完整"

def check_m2m100_model_integrity(local_dir):
    """检查M2M-100模型完整性"""
    # 基本必要文件
    base_files = [
        "config.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json"
    ]
    
    # 分词器模型文件（可能是sentencepiece.bpe.model或其他形式）
    tokenizer_files = [
        "sentencepiece.bpe.model",
        "tokenizer.json",
        "source_spm",
        "target_spm"
    ]
    
    # 检查基本必要文件
    missing_files = []
    for file_name in base_files:
        file_path = os.path.join(local_dir, file_name)
        if not os.path.exists(file_path):
            missing_files.append(file_name)
    
    if missing_files:
        reason = f"缺少必要文件: {', '.join(missing_files)}"
        log_message(f"✗ {reason}", "ERROR")
        return False, reason
    
    # 检查是否至少有一个分词器模型文件
    tokenizer_found = False
    for file_name in tokenizer_files:
        file_path = os.path.join(local_dir, file_name)
        if os.path.exists(file_path):
            tokenizer_found = True
            log_message(f"✓ 找到分词器模型文件: {file_name}")
            break
    
    if not tokenizer_found:
        reason = "缺少分词器模型文件 (sentencepiece.bpe.model、tokenizer.json、source_spm 或 target_spm)"
        log_message(f"✗ {reason}", "ERROR")
        return False, reason
    
    # 检查模型权重文件
    weight_files = [
        "model.safetensors",
        "pytorch_model.bin"
    ]
    
    weight_found = False
    for weight_file in weight_files:
        weight_path = os.path.join(local_dir, weight_file)
        if os.path.exists(weight_path):
            weight_found = True
            # 检查文件大小
            file_size = os.path.getsize(weight_path)
            # M2M-100模型至少应该有几百MB
            if file_size < 100 * 1024 * 1024:  # 100MB
                reason = f"模型权重文件过小: {weight_file} ({file_size / (1024*1024):.2f}MB)"
                log_message(f"✗ {reason}", "ERROR")
                return False, reason
            log_message(f"✓ 找到模型权重文件: {weight_file} ({file_size / (1024*1024):.2f}MB)")
            break
    
    if not weight_found:
        reason = "未找到模型权重文件 (model.safetensors 或 pytorch_model.bin)"
        log_message(f"✗ {reason}", "ERROR")
        return False, reason
    
    log_message("✓ M2M-100模型完整性检查通过", "SUCCESS")
    return True, "模型文件完整"

def check_generic_model_integrity(local_dir):
    """通用模型完整性检查"""
    # 检查是否有配置文件
    config_path = os.path.join(local_dir, "config.json")
    if not os.path.exists(config_path):
        reason = "缺少config.json配置文件"
        log_message(f"✗ {reason}", "ERROR")
        return False, reason
    
    # 检查是否有权重文件
    weight_files = ["model.safetensors", "pytorch_model.bin"]
    weight_found = False
    
    for weight_file in weight_files:
        weight_path = os.path.join(local_dir, weight_file)
        if os.path.exists(weight_path):
            weight_found = True
            log_message(f"✓ 找到模型权重文件: {weight_file}")
            break
    
    if not weight_found:
        reason = "未找到模型权重文件"
        log_message(f"✗ {reason}", "ERROR")
        return False, reason
    
    log_message("✓ 通用模型完整性检查通过", "SUCCESS")
    return True, "模型文件完整"

def check_model_version_compatibility(model_id, local_dir):
    """检查模型版本兼容性"""
    log_message(f"检查模型版本兼容性: {model_id}")
    
    # 检查配置文件
    config_path = os.path.join(local_dir, "config.json")
    if not os.path.exists(config_path):
        reason = "缺少config.json配置文件，无法检查版本"
        log_message(f"✗ {reason}", "ERROR")
        return False, reason
    
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        
        # 提取版本信息
        if "_name_or_path" in config:
            model_name = config["_name_or_path"]
            log_message(f"✓ 模型名称: {model_name}")
        
        if "version" in config:
            version = config["version"]
            log_message(f"✓ 模型版本: {version}")
        
        # 检查模型架构
        if "model_type" in config:
            model_type = config["model_type"]
            log_message(f"✓ 模型架构: {model_type}")
        
        log_message("✓ 模型版本兼容性检查通过", "SUCCESS")
        return True, "模型版本兼容"
    except Exception as e:
        reason = f"版本检查失败: {str(e)}"
        log_message(f"✗ {reason}", "ERROR")
        return False, reason

def check_local_cache(model_id):
    """检查本地缓存"""
    log_message(f"检查本地缓存: {model_id}")
    
    try:
        # 确保导入必要的模块
        import transformers
        import os
        
        try:
            # 尝试不同的API路径
            if hasattr(transformers.utils, 'hub') and hasattr(transformers.utils.hub, 'get_cache_dir'):
                cache_dir = transformers.utils.hub.get_cache_dir()
            elif hasattr(transformers.utils, 'get_cache_dir'):
                cache_dir = transformers.utils.get_cache_dir()
            else:
                # 使用huggingface_hub获取缓存目录
                from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
                cache_dir = HUGGINGFACE_HUB_CACHE
        except Exception:
            # 如果所有方法都失败，使用默认缓存目录
            home_dir = os.path.expanduser("~")
            cache_dir = os.path.join(home_dir, ".cache", "huggingface", "hub")
        
        log_message(f"Transformers缓存目录: {cache_dir}")
        
        # 构建缓存路径
        model_name = model_id.replace("/", "--")
        cache_path = os.path.join(cache_dir, "models--" + model_name)
        
        if os.path.exists(cache_path):
            log_message(f"✓ 发现本地缓存: {cache_path}")
            return True, "本地缓存可用"
        else:
            log_message(f"✗ 未找到本地缓存: {cache_path}")
            return False, "本地缓存不可用"
    except Exception as e:
        log_message(f"✗ 本地缓存检查失败: {str(e)}", "ERROR")
        return False, "本地缓存检查失败"

def should_skip_download(model_id, local_dir):
    """判断是否应该跳过下载"""
    log_message(f"判断是否跳过下载: {model_id}")
    
    # 检查模型文件完整性
    integrity_ok, integrity_reason = check_model_integrity(model_id, local_dir)
    
    if integrity_ok:
        # 检查版本兼容性
        version_ok, version_reason = check_model_version_compatibility(model_id, local_dir)
        
        if version_ok:
            # 检查本地缓存
            cache_ok, cache_reason = check_local_cache(model_id)
            
            log_message(f"✓ 跳过下载: {model_id}", "SUCCESS")
            log_message(f"原因: {integrity_reason}, {version_reason}")
            return True
    
    log_message(f"✗ 需要下载: {model_id}")
    log_message(f"原因: {integrity_reason}")
    return False

def check_internet_connection():
    """检查网络连接"""
    try:
        response = requests.get("https://hf-mirror.com", timeout=10)
        return True
    except:
        return False

def cleanup_duplicate_files(directory):
    """清理目录中的重复文件（带有数字后缀的文件）"""
    if not os.path.exists(directory):
        return
    
    log_message(f"清理目录中的重复文件: {directory}")
    
    # 获取目录中的所有文件
    files = os.listdir(directory)
    
    # 分组文件，按原始文件名分组
    file_groups = {}
    for file in files:
        # 检查是否是重复文件（带有数字后缀）
        import re
        match = re.match(r'^(.*)\.(\d+)\.(.*)$', file)
        if match:
            base_name = match.group(1)
            ext = match.group(3)
            original_file = f"{base_name}.{ext}"
            if original_file not in file_groups:
                file_groups[original_file] = []
            file_groups[original_file].append(file)
    
    # 删除重复文件
    deleted_count = 0
    for original_file, duplicate_files in file_groups.items():
        original_path = os.path.join(directory, original_file)
        # 检查原始文件是否存在
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
        log_message(f"✓ 成功删除 {deleted_count} 个重复文件")
    else:
        log_message("✓ 没有发现重复文件")

def get_file_info(file_path):
    """获取文件的信息，包括大小和修改时间"""
    if not os.path.exists(file_path):
        return None
    
    # 跳过隐藏文件和临时文件
    file_name = os.path.basename(file_path)
    if file_name.startswith('.') or file_name.endswith('.aria2'):
        return None
    
    try:
        file_size = os.path.getsize(file_path)
        file_mtime = os.path.getmtime(file_path)
        return {
            "size": file_size,
            "mtime": file_mtime
        }
    except Exception as e:
        log_message(f"✗ 获取文件信息失败: {str(e)}", "ERROR")
        return None

def load_file_info(model_dir):
    """从内存缓存中加载文件信息"""
    global file_info_cache
    if model_dir in file_info_cache:
        log_message(f"✓ 从内存缓存加载文件信息")
        return file_info_cache[model_dir]
    else:
        return {}

def save_file_info(model_dir, file_info):
    """将文件信息保存到内存缓存"""
    global file_info_cache
    file_info_cache[model_dir] = file_info
    log_message(f"✓ 文件信息保存到内存缓存")

def is_file_intact(file_path, expected_info):
    """检查文件是否完整"""
    if not os.path.exists(file_path):
        return False
    
    current_info = get_file_info(file_path)
    if not current_info:
        return False
    
    # 检查文件大小是否匹配
    if "size" in expected_info and current_info["size"] != expected_info["size"]:
        return False
    
    return True

def update_file_info(model_dir):
    """更新模型目录的文件信息"""
    if not os.path.exists(model_dir):
        return {}
    
    log_message(f"更新文件信息: {model_dir}")
    
    file_info = {}
    for root, dirs, files in os.walk(model_dir):
        # 跳过隐藏目录
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        for file in files:
            # 跳过隐藏文件和临时文件
            if file.startswith('.') or file.endswith('.aria2'):
                continue
            
            file_path = os.path.join(root, file)
            # 计算相对路径作为键
            rel_path = os.path.relpath(file_path, model_dir)
            info = get_file_info(file_path)
            if info:
                file_info[rel_path] = info
    
    # 保存文件信息
    save_file_info(model_dir, file_info)
    return file_info

# 初始化文件信息缓存
initialize_file_info_cache()

def check_file_integrity_with_info(model_dir):
    """使用文件信息检查模型完整性"""
    log_message(f"使用文件信息检查完整性: {model_dir}")
    
    # 加载保存的文件信息
    saved_info = load_file_info(model_dir)
    
    if not saved_info:
        log_message("✗ 没有找到保存的文件信息")
        return False
    
    # 检查每个文件
    all_intact = True
    for rel_path, expected_info in saved_info.items():
        file_path = os.path.join(model_dir, rel_path)
        if not is_file_intact(file_path, expected_info):
            log_message(f"✗ 文件不完整: {rel_path}")
            all_intact = False
    
    if all_intact:
        log_message("✓ 所有文件都完整")
    else:
        log_message("✗ 部分文件不完整")
    
    return all_intact

def download_and_install_aria2():
    """从阿里云镜像下载并安装aria2"""
    log_message("开始下载和安装aria2...")
    
    try:
        import os
        import urllib.request
        import zipfile
        import shutil
        
        aria2_dir = os.path.join(os.getcwd(), "aria2")
        os.makedirs(aria2_dir, exist_ok=True)
        
        # 下载源列表
        sources = [
            ("GitHub", "https://github.com/aria2/aria2/releases/download/release-1.37.0/aria2-1.37.0-win-64bit-build1.zip"),
            ("阿里云镜像", "https://mirrors.aliyun.com/cygwin/x86_64/release/aria2/aria2-1.37.0-1.tar.xz"),
        ]
        
        success = False
        for source_name, url in sources:
            log_message(f"尝试从 {source_name} 下载aria2...")
            
            zip_path = os.path.join(aria2_dir, "aria2.zip")
            
            try:
                # 下载文件
                log_message(f"下载地址: {url}")
                
                def report_hook(count, block_size, total_size):
                    if total_size > 0:
                        percent = int(count * block_size * 100 / total_size)
                        log_message(f"下载进度: {percent}%")
                
                urllib.request.urlretrieve(url, zip_path, report_hook)
                
                if os.path.exists(zip_path):
                    log_message(f"下载完成，正在解压...")
                    
                    # 检查文件类型并解压
                    if zip_path.endswith('.zip'):
                        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                            zip_ref.extractall(aria2_dir)
                    elif zip_path.endswith('.tar.xz'):
                        # 对于tar.xz需要额外处理
                        import tarfile
                        with tarfile.open(zip_path, 'r:xz') as tar:
                            tar.extractall(aria2_dir)
                    
                    # 删除压缩包
                    os.remove(zip_path)
                    
                    # 查找aria2c.exe
                    aria2_exe = None
                    for root, dirs, files in os.walk(aria2_dir):
                        for file in files:
                            if file == "aria2c.exe":
                                aria2_exe = os.path.join(root, file)
                                break
                        if aria2_exe:
                            break
                    
                    if aria2_exe:
                        log_message(f"✓ aria2安装成功: {aria2_exe}")
                        success = True
                        break
                    else:
                        log_message("✗ 未找到aria2c.exe", "ERROR")
            except Exception as e:
                log_message(f"✗ 从 {source_name} 下载失败: {str(e)}", "ERROR")
                if os.path.exists(zip_path):
                    os.remove(zip_path)
        
        if success:
            return True
        else:
            log_message("✗ aria2安装失败", "ERROR")
            return False
            
    except Exception as e:
        log_message(f"✗ aria2安装过程出错: {str(e)}", "ERROR")
        return False

def check_aria2_installed():
    """检查是否安装了aria2，如果没有则尝试安装"""
    try:
        import subprocess
        import os
        
        # 首先检查系统PATH中的aria2c
        result = subprocess.run(
            "aria2c --version",
            shell=True,
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            log_message("✓ 使用系统PATH中的aria2")
            return True
        
        # 然后检查当前目录下的aria2（各种可能的路径）
        aria2_dir = os.path.join(os.getcwd(), "aria2")
        if os.path.exists(aria2_dir):
            for root, dirs, files in os.walk(aria2_dir):
                for file in files:
                    if file == "aria2c.exe":
                        aria2_path = os.path.join(root, file)
                        result = subprocess.run(
                            f'"{aria2_path}" --version',
                            shell=True,
                            capture_output=True,
                            text=True
                        )
                        if result.returncode == 0:
                            log_message(f"✓ 使用本地aria2: {aria2_path}")
                            return True
        
        # 如果都没找到，尝试下载安装
        log_message("aria2未安装，尝试自动下载安装...")
        if download_and_install_aria2():
            # 再次检查
            return check_aria2_installed()
        else:
            return False
        
    except Exception as e:
        log_message(f"检查aria2时出错: {str(e)}", "ERROR")
        return False

def get_optimal_threads():
    """根据系统配置获取最佳线程数"""
    try:
        import os
        import multiprocessing
        
        # 获取CPU核心数
        cpu_count = multiprocessing.cpu_count()
        log_message(f"检测到CPU核心数: {cpu_count}")
        
        # 根据CPU核心数计算最佳线程数
        # 通常线程数是CPU核心数的2倍，最大不超过16（aria2的限制）
        optimal_threads = max(8, min(16, cpu_count * 2))
        log_message(f"计算最佳线程数: {optimal_threads}")
        
        return optimal_threads
    except Exception as e:
        log_message(f"获取线程数失败: {str(e)}", "ERROR")
        # 默认使用16线程
        return 16

def find_aria2_path():
    """查找aria2c.exe的路径"""
    import os
    
    # 1. 检查系统PATH
    try:
        import subprocess
        result = subprocess.run("aria2c --version", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            return "aria2c"
    except:
        pass
    
    # 2. 检查当前目录下的aria2目录
    aria2_dir = os.path.join(os.getcwd(), "aria2")
    if os.path.exists(aria2_dir):
        for root, dirs, files in os.walk(aria2_dir):
            for file in files:
                if file == "aria2c.exe":
                    return os.path.join(root, file)
    
    # 3. 检查特定路径
    specific_paths = [
        os.path.join(os.getcwd(), "aria2", "aria2-1.37.0-win-64bit-build1", "aria2c.exe"),
    ]
    for path in specific_paths:
        if os.path.exists(path):
            return path
    
    return None

def download_with_aria2(model_id, local_dir):
    """使用aria2进行多线程下载"""
    log_message(f"使用aria2多线程下载模型: {model_id}")
    
    try:
        import subprocess
        import os
        from huggingface_hub import HfApi
        
        # 查找aria2路径
        aria2_path = find_aria2_path()
        if not aria2_path:
            log_message("✗ 未找到aria2c.exe", "ERROR")
            return False
        
        log_message(f"使用aria2: {aria2_path}")
        
        # 获取模型文件列表
        api = HfApi(endpoint=os.environ["HF_ENDPOINT"])
        files = api.list_repo_files(repo_id=model_id)
        
        # 打印所有找到的文件
        log_message(f"找到 {len(files)} 个文件:")
        for file in files:
            log_message(f"  - {file}")
        
        # 只下载必要文件
        required_files = []
        for file in files:
            if any(file.endswith(ext) for ext in [".bin", ".safetensors", ".json", ".spm", ".model", ".txt"]):
                required_files.append(file)
        
        log_message(f"需要下载 {len(required_files)} 个文件:")
        for file in required_files:
            log_message(f"  - {file}")
        
        # 下载每个文件
        for file in required_files:
            file_path = os.path.join(local_dir, file)
            file_dir = os.path.dirname(file_path)
            os.makedirs(file_dir, exist_ok=True)
            
            # 检查文件是否已存在且完整
            if os.path.exists(file_path):
                # 检查文件大小，确保文件不是空的
                file_size = os.path.getsize(file_path)
                if file_size > 0:
                    # 加载保存的文件信息
                    saved_info = load_file_info(local_dir)
                    rel_path = os.path.relpath(file_path, local_dir)
                    
                    if rel_path in saved_info and is_file_intact(file_path, saved_info[rel_path]):
                        log_message(f"✓ 文件已存在且完整，跳过下载: {file}")
                        # 清理可能的重复文件
                        cleanup_duplicate_files(file_dir)
                        continue
                    else:
                        log_message(f"✗ 文件存在但不完整，重新下载: {file}")
            
            # 构建下载URL
            url = f"{os.environ['HF_ENDPOINT']}/{model_id}/resolve/main/{file}"
            log_message(f"下载文件: {file}")
            
            # 获取最佳线程数
            threads = get_optimal_threads()
            
            # 切换到目标目录的父目录，使用相对路径避免路径拼接错误
            file_dir = os.path.dirname(file_path)
            os.makedirs(file_dir, exist_ok=True)
            file_name = os.path.basename(file_path)
            
            # 构建aria2命令
            if aria2_path == "aria2c":
                # 使用系统PATH中的aria2
                command = f'cd "{file_dir}" && aria2c -x {threads} -s {threads} -o "{file_name}" "{url}"'
            else:
                # 使用指定路径的aria2
                command = f'cd "{file_dir}" && "{aria2_path}" -x {threads} -s {threads} -o "{file_name}" "{url}"'
            
            log_message(f"使用aria2多线程下载，{threads}线程")
            log_message(f"执行命令: {command}")
            
            # 执行命令，实时显示进度
            result = subprocess.run(
                command,
                shell=True,
                capture_output=False,
                text=True,
                timeout=3600  # 1小时超时
            )
            
            # 检查执行结果
            if result.returncode == 0:
                log_message(f"✓ 文件下载完成: {file}")
                # 清理可能的重复文件
                cleanup_duplicate_files(file_dir)
                # 更新文件信息
                update_file_info(local_dir)
            else:
                log_message(f"✗ 文件下载失败: {file}", "ERROR")
                return False
        
        # 下载完成后，清理整个模型目录中的重复文件
        log_message("清理模型目录中的重复文件...")
        cleanup_duplicate_files(local_dir)
        
        # 更新文件信息
        log_message("更新模型文件信息...")
        update_file_info(local_dir)
        
        # 检查模型完整性
        log_message("检查模型完整性...")
        integrity_ok, integrity_reason = check_model_integrity(model_id, local_dir)
        
        if integrity_ok:
            log_message(f"✓ 模型完整性检查通过: {local_dir}", "SUCCESS")
            return True
        else:
            log_message(f"✗ 下载后模型完整性检查失败: {integrity_reason}", "ERROR")
            return False
            
    except Exception as e:
        log_message(f"✗ aria2下载失败: {str(e)}", "ERROR")
        return False

def download_with_curl(model_id, local_dir):
    """使用curl命令直接下载模型"""
    log_message(f"使用curl命令直接下载模型: {model_id}")
    
    try:
        import subprocess
        from huggingface_hub import HfApi
        
        # 获取模型文件列表
        api = HfApi(endpoint=os.environ["HF_ENDPOINT"])
        files = api.list_repo_files(repo_id=model_id)
        
        # 只下载必要文件
        required_files = []
        for file in files:
            if any(file.endswith(ext) for ext in [".bin", ".safetensors", ".json", ".spm", ".model"]):
                required_files.append(file)
        
        log_message(f"需要下载 {len(required_files)} 个文件")
        
        # 下载每个文件
        for file in required_files:
            file_path = os.path.join(local_dir, file)
            file_dir = os.path.dirname(file_path)
            os.makedirs(file_dir, exist_ok=True)
            
            # 检查文件是否已存在且完整
            if os.path.exists(file_path):
                # 检查文件大小，确保文件不是空的
                file_size = os.path.getsize(file_path)
                if file_size > 0:
                    # 加载保存的文件信息
                    saved_info = load_file_info(local_dir)
                    rel_path = os.path.relpath(file_path, local_dir)
                    
                    if rel_path in saved_info and is_file_intact(file_path, saved_info[rel_path]):
                        log_message(f"✓ 文件已存在且完整，跳过下载: {file}")
                        # 清理可能的重复文件
                        cleanup_duplicate_files(file_dir)
                        continue
                    else:
                        log_message(f"✗ 文件存在但不完整，重新下载: {file}")
            
            # 构建下载URL
            url = f"{os.environ['HF_ENDPOINT']}/{model_id}/resolve/main/{file}"
            log_message(f"下载文件: {file}")
            
            # 构建curl命令（Windows专用）
            command = f'curl -L -o "{file_path}" "{url}" --progress-bar'
            log_message("使用curl命令下载，显示详细进度条")
            
            log_message(f"执行命令: {command}")
            
            # 执行命令，实时显示进度条
            result = subprocess.run(
                command,
                shell=True,
                capture_output=False,  # 不捕获输出，让进度条实时显示
                text=True,
                timeout=3600  # 1小时超时
            )
            
            # 检查执行结果
            if result.returncode == 0:
                log_message(f"✓ 文件下载完成: {file}")
                # 清理可能的重复文件
                cleanup_duplicate_files(file_dir)
                # 更新文件信息
                update_file_info(local_dir)
            else:
                log_message(f"✗ 文件下载失败: {file}", "ERROR")
                return False
        
        # 下载完成后，清理整个模型目录中的重复文件
        log_message("清理模型目录中的重复文件...")
        cleanup_duplicate_files(local_dir)
        
        # 更新文件信息
        log_message("更新模型文件信息...")
        update_file_info(local_dir)
        
        # 检查模型完整性
        log_message("检查模型完整性...")
        integrity_ok, integrity_reason = check_model_integrity(model_id, local_dir)
        
        if integrity_ok:
            log_message(f"✓ 模型完整性检查通过: {local_dir}", "SUCCESS")
            return True
        else:
            log_message(f"✗ 下载后模型完整性检查失败: {integrity_reason}", "ERROR")
            return False
            
    except Exception as e:
        log_message(f"✗ curl下载失败: {str(e)}", "ERROR")
        return False

def test_model_usability(model_id, local_dir):
    """测试模型是否能使用"""
    log_message(f"\n=== 测试模型可用性: {model_id} ===", "INFO")
    
    # 首先检查模型完整性
    integrity_ok, integrity_msg = check_model_integrity(model_id, local_dir)
    
    if not integrity_ok:
        log_message(f"✗ 模型完整性检查失败: {integrity_msg}", "ERROR")
        return False
    
    log_message("✓ 模型完整性检查通过")
    
    # 尝试导入必要的模块进行功能测试
    if "whisper" in model_id.lower():
        try:
            from faster_whisper import WhisperModel
            import os
            
            # 尝试加载模型（不实际运行推理）
            log_message("尝试加载Whisper模型...")
            model = WhisperModel(local_dir, device="cpu")
            log_message("✓ Whisper模型加载成功")
            return True
        except Exception as e:
            log_message(f"✗ Whisper模型加载失败: {str(e)}", "ERROR")
            return False
    elif "m2m100" in model_id.lower():
        try:
            from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
            import torch
            
            # 尝试加载模型（不实际运行推理）
            log_message("尝试加载M2M100模型...")
            tokenizer = M2M100Tokenizer.from_pretrained(local_dir)
            model = M2M100ForConditionalGeneration.from_pretrained(local_dir)
            log_message("✓ M2M100模型加载成功")
            return True
        except Exception as e:
            log_message(f"✗ M2M100模型加载失败: {str(e)}", "ERROR")
            return False
    else:
        log_message("✓ 模型类型未知，仅通过完整性检查")
        return True

def download_model_with_retry(model_id, local_dir, max_retries=3):
    """下载模型并重试直到成功"""
    retry_count = 0
    
    while retry_count < max_retries:
        log_message(f"\n=== 尝试下载模型: {model_id} (尝试 {retry_count+1}/{max_retries}) ===", "INFO")
        
        if download_model_with_hfd(model_id, local_dir):
            # 测试模型是否能使用
            if test_model_usability(model_id, local_dir):
                log_message(f"✓ 模型下载和测试成功: {model_id}")
                return True
            else:
                log_message(f"✗ 模型测试失败，准备重新下载: {model_id}", "ERROR")
        else:
            log_message(f"✗ 模型下载失败，准备重新下载: {model_id}", "ERROR")
        
        retry_count += 1
        if retry_count < max_retries:
            log_message(f"等待5秒后重试...")
            import time
            time.sleep(5)
    
    log_message(f"✗ 模型下载失败，已达到最大重试次数: {model_id}", "ERROR")
    return False

def download_model_with_hfd(model_id, local_dir, max_retries=5):
    """使用curl下载模型（Windows专用）"""
    log_message(f"\n下载模型: {model_id}")
    log_message(f"本地目录: {local_dir}")
    
    # 创建本地目录
    os.makedirs(local_dir, exist_ok=True)
    
    # 检查是否应该跳过下载
    if should_skip_download(model_id, local_dir):
        # 为已存在的模型生成文件信息
        log_message(f"✓ 跳过下载: {model_id}")
        log_message("生成文件信息...")
        update_file_info(local_dir)
        return True
    
    # 检查网络连接
    if not check_internet_connection():
        log_message("✗ 网络连接失败，请检查网络设置", "ERROR")
        return False
    
    # Windows系统，优先使用aria2进行多线程下载
    if check_aria2_installed():
        log_message("Windows系统，使用aria2多线程下载模型")
        return download_with_aria2(model_id, local_dir)
    else:
        log_message("Windows系统，aria2未安装，使用curl下载模型")
        return download_with_curl(model_id, local_dir)

def main():
    """下载所有需要的模型"""
    log_message("开始下载所有模型...", "INFO")
    
    # 定义需要下载的模型
    models = [
        # Faster-Whisper系列模型 - 不同大小的版本
        {
            "id": "Systran/faster-whisper-tiny",
            "dir": os.path.join(MODEL_DIR, "Systran--faster-whisper-tiny"),
            "priority": "medium"
        },
        {
            "id": "Systran/faster-whisper-base",
            "dir": os.path.join(MODEL_DIR, "Systran--faster-whisper-base"),
            "priority": "medium"
        },
        {
            "id": "Systran/faster-whisper-small",
            "dir": os.path.join(MODEL_DIR, "Systran--faster-whisper-small"),
            "priority": "medium"
        },
        {
            "id": "Systran/faster-whisper-medium",
            "dir": os.path.join(MODEL_DIR, "Systran--faster-whisper-medium"),
            "priority": "medium"
        },
        {
            "id": "Systran/faster-whisper-large-v2",
            "dir": os.path.join(MODEL_DIR, "Systran--faster-whisper-large-v2"),
            "priority": "high"
        },
        {
            "id": "Systran/faster-whisper-large-v3",
            "dir": os.path.join(MODEL_DIR, "Systran--faster-whisper-large-v3"),
            "priority": "high"
        },
        # M2M-100翻译模型系列
        {
            "id": "facebook/m2m100_418M",
            "dir": os.path.join(MODEL_DIR, "facebook--m2m100_418M"),
            "priority": "medium"
        },
        {
            "id": "facebook/m2m100_1.2B",
            "dir": os.path.join(MODEL_DIR, "facebook--m2m100_1.2B"),
            "priority": "high"
        }
    ]
    
    # 按照优先级排序，高优先级的模型先下载
    models.sort(key=lambda x: 0 if x["priority"] == "high" else 1)
    
    # 下载所有模型（带重试和测试）
    success_count = 0
    total_count = len(models)
    
    for model in models:
        log_message(f"\n=== 处理模型: {model['id']} ===")
        log_message(f"优先级: {model['priority']}")
        
        if download_model_with_retry(model["id"], model["dir"]):
            success_count += 1
            log_message(f"✓ 模型处理成功: {model['id']}")
        else:
            log_message(f"✗ 模型处理失败: {model['id']}", "ERROR")
    
    log_message(f"\n===== 下载完成 ====\n")
    log_message(f"成功下载: {success_count}/{total_count} 个模型")
    
    # 保存所有模型信息到根目录的.file_info.json
    save_all_file_info()
    
    if success_count == total_count:
        log_message("✓ 所有模型下载成功且测试通过！")
    else:
        log_message("✗ 部分模型下载失败，请稍后再试", "ERROR")
        log_message("\n提示: 您也可以手动下载模型并放入对应的目录中:")
        for model in models:
            log_message(f"  - {model['id']} 到 {model['dir']}")

if __name__ == "__main__":
    main()