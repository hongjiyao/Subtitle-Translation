import os

# 确保在导入模型库之前设置HF-Mirror作为下载源
# 这些环境变量需要在导入其他库之前设置
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HUGGINGFACE_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.environ["PYTORCH_TRANSFORMERS_CACHE"] = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

# 尝试导入faster-whisper，如果失败则使用原始whisper
FASTER_WHISPER_AVAILABLE = False
try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    import whisper

from config import MODEL_CACHE_DIR

# 模型缓存，用于存储已加载的模型实例
MODEL_CACHE = {}

# 内存使用监控（简化版，不依赖psutil）
import os
import gc

def clear_model_cache():
    """清空模型缓存以释放内存"""
    global MODEL_CACHE
    if MODEL_CACHE:
        print(f"[内存管理] 清空模型缓存，释放内存")
        # 逐个删除模型实例，确保完全释放
        for key in list(MODEL_CACHE.keys()):
            del MODEL_CACHE[key]
        MODEL_CACHE = {}
        # 强制垃圾回收
        gc.collect()
        # 再次垃圾回收，确保完全释放
        gc.collect()
        print(f"[内存管理] 已执行垃圾回收")

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
                            print(f"[语音识别] 找到本地模型快照: {snapshot_path}")
                            return snapshot_path
                except:
                    pass
            elif os.path.isdir(path):
                print(f"[语音识别] 找到本地模型: {path}")
                return path
    
    return None

def recognize_speech(audio_path, model_path, detected_language=None, device_choice="auto", progress_callback=None, beam_size=1, vad_filter=True, word_timestamps=False, condition_on_previous_text=False):
    """使用Whisper或Faster-Whisper模型进行语音识别"""
    if FASTER_WHISPER_AVAILABLE:
        # 使用Faster-Whisper模型
        device = "cuda" if device_choice != "cpu" else "cpu"
        cache_key = f"faster-whisper-{model_path}-{device}"
        
        # 检查模型是否已在缓存中
        if cache_key not in MODEL_CACHE:
            # 检查本地模型
            local_model_path = check_local_model(model_path)
            
            if local_model_path:
                # 使用本地模型
                model = WhisperModel(local_model_path, device=device)
            else:
                # 只使用本地模型，不下载
                error_msg = f"本地模型不存在: {model_path}，请确保模型已在models目录中"
                print(f"[错误信息] {error_msg}")
                raise FileNotFoundError(error_msg)
            
            # 将模型添加到缓存
            MODEL_CACHE[cache_key] = model
        
        model = MODEL_CACHE[cache_key]
        
        # 转换结果为与原始Whisper兼容的格式
        result = {
            'text': '',
            'segments': [],
            'language': ''
        }
        
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
        
        # 使用Faster-Whisper进行语音识别
        print("[语音识别] 开始语音识别...")
        print(f"[语音识别] 音频时长: {audio_duration:.2f} 秒")
        print(f"[语音识别] 使用模型: {model_path}")
        print(f"[语音识别] 设备: {device}")
        
        segments, info = model.transcribe(
            audio_path, 
            beam_size=beam_size,
            language=detected_language if detected_language else None,
            vad_filter=vad_filter,
            word_timestamps=word_timestamps,
            condition_on_previous_text=condition_on_previous_text,
            vad_parameters={"threshold": 0.5, "min_silence_duration_ms": 500}
        )
        
        print(f"[语音识别] 检测到的语言: {info.language}")
        print("[语音识别] 开始处理识别结果...")
        
        result['language'] = info.language
        current_time = 0
        
        # 处理每个片段并更新进度
        for segment in segments:
            result['text'] += segment.text
            result['segments'].append({
                'id': len(result['segments']),
                'seek': int(segment.start * 1000),
                'start': segment.start,
                'end': segment.end,
                'text': segment.text,
                'tokens': [],
                'temperature': 0.0,
                'avg_logprob': 0.0,
                'compression_ratio': 0.0,
                'no_speech_prob': 0.0
            })
            
            # 更新当前处理时间
            current_time = segment.end
            
            # 计算进度并更新
            if progress_callback and audio_duration > 0:
                progress = min(int((current_time / audio_duration) * 100), 95)
                progress_callback(progress)
        
        # 完成进度条
        if progress_callback:
            progress_callback(100)
        
        # 保留模型缓存，避免重复加载
        return result
    else:
        # 使用原始Whisper模型
        device = "cpu" if device_choice == "cpu" else "cuda"
        cache_key = f"whisper-{model_path}-{device}"
        
        # 检查模型是否已在缓存中
        if cache_key not in MODEL_CACHE:
            # 检查本地模型
            local_model_path = check_local_model(model_path)
            
            # 加载模型
            if local_model_path:
                if device_choice == "cpu":
                    model = whisper.load_model(local_model_path, device="cpu")
                else:
                    model = whisper.load_model(local_model_path)
            else:
                # 只使用本地模型，不下载
                error_msg = f"本地模型不存在: {model_path}，请确保模型已在models目录中"
                print(f"[错误信息] {error_msg}")
                raise FileNotFoundError(error_msg)
            
            # 将模型添加到缓存
            MODEL_CACHE[cache_key] = model
        
        model = MODEL_CACHE[cache_key]
        
        # 识别音频
        if detected_language:
            result = model.transcribe(audio_path, language=detected_language, verbose=False)
        else:
            # 自动检测语言
            result = model.transcribe(audio_path, verbose=False)
        
        # 更新进度条
        if progress_callback:
            progress_callback(100)
        
        # 保留模型缓存，避免重复加载
        return result
