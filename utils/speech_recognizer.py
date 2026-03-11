import os
import datetime

# 确保在导入模型库之前设置HF-Mirror作为下载源
# 这些环境变量需要在导入其他库之前设置
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HUGGINGFACE_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.environ["HF_HOME"] = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

# 获取带时间戳的打印函数
def timestamp_print(message):
    """带时间戳的打印函数"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

# 导入WhisperX（必需，用于语音活动检测和批处理功能）
try:
    # 在导入whisperx之前设置TF32，解决pyannote.audio的ReproducibilityWarning
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    import whisperx
    WHISPERX_AVAILABLE = True
    # 抑制 torchcodec 警告和ReproducibilityWarning
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio.core.io")
    warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio.utils.reproducibility")
except ImportError:
    timestamp_print("[错误] WhisperX 未安装，请安装后再使用")
    raise

from config import MODEL_CACHE_DIR

# 内存使用监控（简化版，不依赖psutil）
import gc

def clear_model_cache():
    """清空模型缓存以释放内存"""
    # 强制垃圾回收
    gc.collect()
    # 清空CUDA缓存
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        timestamp_print(f"[内存管理] 清空CUDA缓存时出错: {str(e)}")
    # 再次垃圾回收，确保完全释放
    gc.collect()
    timestamp_print("[内存管理] 已执行垃圾回收")

# 检查本地模型文件是否存在
def check_local_model(model_name):
    """检查本地模型文件是否存在"""
    model_paths = [
        # 直接在models目录下
        os.path.join(MODEL_CACHE_DIR, model_name),
        # 常见的命名格式
        os.path.join(MODEL_CACHE_DIR, f"whisper-{model_name}"),
        os.path.join(MODEL_CACHE_DIR, f"openai--whisper-{model_name}"),
        os.path.join(MODEL_CACHE_DIR, f"Systran--faster-whisper-{model_name}"),
        # 快照目录格式
        os.path.join(MODEL_CACHE_DIR, f"models--openai--whisper-{model_name}", "snapshots"),
        os.path.join(MODEL_CACHE_DIR, f"models--Systran--faster-whisper-{model_name}", "snapshots"),
        # 子目录格式
        os.path.join(MODEL_CACHE_DIR, "openai", f"whisper-{model_name}"),
        os.path.join(MODEL_CACHE_DIR, "Systran", f"faster-whisper-{model_name}")
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            # 如果是快照目录，查找其中的子目录
            if "snapshots" in path and os.path.isdir(path):
                # 查找快照目录中的第一个子目录
                try:
                    snapshot_dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
                    if snapshot_dirs:
                        snapshot_path = os.path.join(path, snapshot_dirs[0])
                        if os.path.exists(snapshot_path):
                            timestamp_print(f"[语音识别] 找到本地模型快照: {snapshot_path}")
                            return snapshot_path
                except:
                    pass
            elif os.path.isdir(path):
                timestamp_print(f"[语音识别] 找到本地模型: {path}")
                return path
    
    return None

def extract_middle_audio(audio_path, duration=30):
    """提取音频中间的指定时长片段"""
    import os
    import tempfile
    import subprocess
    
    # 估算音频长度
    import wave
    import contextlib
    
    def get_audio_duration(audio_path):
        try:
            with contextlib.closing(wave.open(audio_path, 'r')) as f:
                frames = f.getnframes()
                rate = f.getframerate()
                duration = frames / float(rate)
                return duration
        except:
            return 60  # 默认60秒
    
    audio_duration = get_audio_duration(audio_path)
    
    # 计算中间位置
    start_time = max(0, (audio_duration - duration) / 2)
    
    # 创建临时文件
    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    temp_file_path = temp_file.name
    temp_file.close()
    
    # 使用ffmpeg提取中间片段
    try:
        cmd = [
            'ffmpeg', '-i', audio_path,
            '-ss', str(start_time),
            '-t', str(duration),
            '-acodec', 'pcm_s16le',
            '-ar', '16000',
            '-ac', '1',
            '-y',
            temp_file_path
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return temp_file_path
    except Exception as e:
        timestamp_print(f"[错误信息] 提取中间音频片段时出错: {str(e)}")
        # 如果提取失败，返回原始音频路径
        try:
            os.unlink(temp_file_path)
        except:
            pass
        return audio_path

def recognize_speech(audio_path, model_path, detected_language=None, device_choice="auto", progress_callback=None, beam_size=1, vad_filter=True, word_timestamps=False, condition_on_previous_text=False, use_whisperx=True, whisperx_batch_size=8, vad_threshold=0.5, vad_min_speech_duration_ms=250, vad_max_speech_duration_s=30, vad_min_silence_duration_ms=100):
    """使用WhisperX模型进行语音识别（仅使用Pyannote VAD）"""
    # 始终使用WhisperX和Pyannote VAD
    timestamp_print("[语音识别] 使用 WhisperX 批处理功能进行语音识别")
    timestamp_print("[VAD参数] 使用Pyannote VAD进行语音活动检测")
    timestamp_print(f"[VAD参数] VAD阈值: {vad_threshold}")
    timestamp_print(f"[VAD参数] 最小语音持续时间: {vad_min_speech_duration_ms}ms")
    timestamp_print(f"[VAD参数] 最大语音持续时间: {vad_max_speech_duration_s}s")
    timestamp_print(f"[VAD参数] 最小沉默持续时间: {vad_min_silence_duration_ms}ms")
    
    # 确定设备
    device = "cuda" if device_choice != "cpu" else "cpu"
    
    # 每次都重新加载模型
    # 检查本地模型
    local_model_path = check_local_model(model_path)
    
    if local_model_path:
        # 使用本地模型
        timestamp_print(f"[模型加载] 正在将 WhisperX 模型 {model_path} 加载到内存中...")
        # 加载WhisperX模型，传递VAD参数
        vad_options = {
            "vad_onset": vad_threshold,
            "vad_offset": vad_threshold - 0.137,  # 保持与默认值的关系
            "chunk_size": 30
        }
        model = whisperx.load_model(
            local_model_path if local_model_path else model_path,
            device=device,
            compute_type="float16" if device == "cuda" else "float32",
            vad_options=vad_options
        )
        timestamp_print(f"[模型加载] WhisperX 模型 {model_path} 已成功加载到内存")
    else:
        # 只使用本地模型，不下载
        error_msg = f"本地模型不存在: {model_path}，请确保模型已在models目录中"
        timestamp_print(f"[错误信息] {error_msg}")
        raise FileNotFoundError(error_msg)
    
    # 检测模型加载后的显存使用情况
    if device == "cuda":
        try:
            import subprocess
            import re
            # 执行nvidia-smi命令
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.total,memory.free', '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                check=True
            )
            # 解析输出
            output = result.stdout.strip()
            if output:
                total, free = map(int, output.split(','))
                used = total - free
                timestamp_print(f"[内存管理] 模型加载后GPU显存: 总内存={total/1024:.2f}GB, 可用内存={free/1024:.2f}GB, 已使用={used/1024:.2f}GB")
        except Exception as e:
            timestamp_print(f"[错误信息] 获取模型加载后GPU内存时出错: {str(e)}")
    
    # 开始语音识别
    timestamp_print("[语音识别] 开始语音识别...")
    
    # 开始语音识别前先更新一次进度
    if progress_callback:
        progress_callback(10)
    
    # 估算音频长度
    import wave
    import contextlib
    
    def get_audio_duration(audio_path):
        try:
            with contextlib.closing(wave.open(audio_path, 'r')) as f:
                frames = f.getnframes()
                rate = f.getframerate()
                duration = frames / float(rate)
                return duration
        except:
            return 60  # 默认60秒
    
    audio_duration = get_audio_duration(audio_path)
    timestamp_print(f"[语音识别] 音频时长: {audio_duration:.2f} 秒")
    timestamp_print(f"[语音识别] 使用模型: {model_path}")
    timestamp_print(f"[语音识别] 设备: {device}")
    timestamp_print(f"[语音识别] WhisperX批处理大小: {whisperx_batch_size}")
    
    # 步骤1: 语音识别
    timestamp_print("[语音识别] 执行语音识别...")
    
    # 如果没有指定语言，使用中间30秒进行语言检测
    if not detected_language:
        timestamp_print("[语音识别] 自动检测语言，使用音频中间30秒进行检测...")
        middle_audio = extract_middle_audio(audio_path, 30)
        # 先使用中间片段检测语言
        lang_result = model.transcribe(
            middle_audio,
            language=None,
            batch_size=whisperx_batch_size
        )
        detected_language = lang_result.get('language', 'unknown')
        timestamp_print(f"[语音识别] 检测到的语言: {detected_language}")
        # 清理临时文件
        if middle_audio != audio_path:
            try:
                import os
                os.unlink(middle_audio)
            except:
                pass
    
    # WhisperX的transcribe方法
    result = model.transcribe(
        audio_path,
        language=detected_language if detected_language else None,
        batch_size=whisperx_batch_size  # 批处理大小
    )
    
    # 语音识别完成后更新进度
    if progress_callback:
        progress_callback(50)
    
    # 语音识别完成后更新进度
    if progress_callback:
        progress_callback(80)
    
    # 获取语言信息
    language = result.get('language', 'unknown')
    # 检查是否在 segments 中包含语言信息
    if language == 'unknown' and result.get('segments'):
        for segment in result['segments']:
            if 'language' in segment:
                language = segment['language']
                break
    # 从 WhisperX 的日志中获取语言信息
    if language == 'unknown':
        # 尝试根据识别结果判断语言
        text = ''.join([segment.get('text', '') for segment in result.get('segments', [])])
        # 简单的语言检测
        if any(char >= '\u3040' and char <= '\u30ff' for char in text):
            language = 'ja'  # 日语
        elif any(char >= '\u4e00' and char <= '\u9fff' for char in text):
            language = 'zh'  # 中文
        elif all(ord(c) < 128 or c.isspace() for c in text):
            language = 'en'  # 英语
    
    timestamp_print(f"[语音识别] 检测到的语言: {language}")
    timestamp_print("[语音识别] 开始处理识别结果...")
    
    # 转换结果为与原始Whisper兼容的格式
    final_result = {
        'text': ''.join([segment.get('text', '') for segment in result.get('segments', [])]),
        'segments': [],
        'language': language
    }
    
    # 处理每个片段
    for i, segment in enumerate(result.get('segments', [])):
        segment_data = {
            'id': i,
            'seek': int(segment['start'] * 1000),
            'start': segment['start'],
            'end': segment['end'],
            'text': segment['text'],
            'tokens': [],
            'temperature': segment.get('temperature', 0.0),
            'avg_logprob': segment.get('avg_logprob', 0.0),
            'compression_ratio': segment.get('compression_ratio', 0.0),
            'no_speech_prob': segment.get('no_speech_prob', 0.0)
        }
        
        final_result['segments'].append(segment_data)
    
    # 完成进度条
    if progress_callback:
        progress_callback(100)
    
    # 删除模型对象，释放内存
    try:
        del model
    except Exception as e:
        timestamp_print(f"[内存管理] 删除模型对象时出错: {str(e)}")
    
    # 执行垃圾回收
    import gc
    gc.collect()
    
    return final_result
