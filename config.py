import os

# 项目根目录
PROJECT_ROOT = os.path.dirname(__file__)

# 模型缓存目录
MODEL_CACHE_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# 临时目录
TEMP_DIR = os.path.join(PROJECT_ROOT, "temp")
os.makedirs(TEMP_DIR, exist_ok=True)

# 输出目录
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 默认配置
DEFAULT_CONFIG = {
    # 语音识别模型 (small, medium, large, large-v2, large-v3)
    "speech_recognition_model": "small",
    
    # 翻译模型
    "translation_model": "facebook/m2m100_418M",
    
    # 设备选择 (auto, cpu, cuda)
    "device": "auto",
    
    # 进度条配置
    "progress_bar_ncols": 80,
    
    # 语音识别参数
    "speech_recognition_params": {
        "beam_size": 1,
        "vad_filter": True,
        "word_timestamps": False,
        "condition_on_previous_text": False
    },
    
    # 翻译参数
    "translation_params": {
        "beam_size": 1,
        "max_length": 256,
        "early_stopping": True
    }
}

# 确保模型缓存目录存在
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
