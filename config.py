# -*- coding: utf-8 -*-
"""
配置管理模块
提供系统化的配置参数定义、验证和管理功能
"""

import os
import json
from typing import Dict, Any, List, Tuple, Optional


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_CACHE_DIR = os.path.join(PROJECT_ROOT, "models")
TEMP_DIR = os.path.join(PROJECT_ROOT, "temp")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
CONFIG_FILE = os.path.join(PROJECT_ROOT, "saved_params.json")

for d in [MODEL_CACHE_DIR, TEMP_DIR, OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)


PARAM_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "model": {
        "default": "large-v3-turbo",
        "options": ["tiny", "base", "small", "medium", "large-v2", "large-v3", "large-v3-turbo"],
        "description": "Whisper 语音识别模型"
    },
    "translator": {
        "default": "tencent/HY-MT1.5-7B-GGUF",
        "options": ["tencent/HY-MT1.5-7B-GGUF"],
        "description": "翻译模型标识"
    },
    "translator_quantization": {
        "default": "Q4_K_M",
        "options": ["auto", "Q4_K_M", "Q6_K", "Q8_0"],
        "description": "翻译模型量化版本选择"
    },
    "device": {
        "default": "auto",
        "options": ["auto", "cuda", "cpu"],
        "description": "计算设备选择"
    },
    "source_language": {
        "default": "ja",
        "options": ["zh", "en", "ja", "ko", "fr", "de", "es", "ru", "ar", "hi", "pt", "it", "nl", "pl"],
        "description": "源语言（语音输入语言）"
    },
    "target_language": {
        "default": "zh",
        "options": ["zh", "en", "ja", "ko", "fr", "de", "es", "ru", "ar", "hi", "pt", "it", "nl", "pl"],
        "description": "目标语言（翻译输出语言）"
    },
    "vad_threshold": {
        "default": 0.2,
        "range": [0.1, 0.9],
        "description": "VAD 语音活动检测阈值"
    },
    "vad_min_speech_duration": {
        "default": 1.0,
        "range": [0.1, 10.0],
        "description": "VAD 最小语音持续时间（秒）"
    },
    "vad_max_speech_duration": {
        "default": 30.0,
        "range": [10.0, 300.0],
        "description": "VAD 最大语音持续时间（秒），超过此长度会强制分割"
    },
    "vad_min_silence_duration": {
        "default": 1.5,
        "range": [0.1, 10.0],
        "description": "VAD 最小静默持续时间（秒）"
    },
    "vad_speech_pad_ms": {
        "default":200,
        "range": [0, 5000],
        "description": "VAD 语音填充时间（毫秒）"
    },
    "vad_prefix_padding_ms": {
        "default":300,
        "range": [0, 5000],
        "description": "VAD 前缀填充时间（毫秒）"
    },
    "vad_neg_threshold": {
        "default": None,
        "range": [0.0, 1.0],
        "description": "VAD 静音判定阈值，默认为 threshold-0.15"
    },
    "use_max_poss_sil_at_max_speech": {
        "default": True,
        "description": "VAD 是否优先使用最长静音分割点"
    },
    "enable_whispercd": {
        "default": True,
        "description": "启用 Whisper-CD 处理器"
    },
    "whispercd_alpha": {
        "default": 1.0,
        "range": [0.0, 2.0],
        "description": "Whisper-CD 对比强度参数"
    },
    "whispercd_temperature": {
        "default": 1.0,
        "range": [0.1, 5.0],
        "description": "Whisper-CD log-sum-exp 温度参数"
    },
    "whispercd_snr_db": {
        "default": 10.0,
        "range": [0.0, 30.0],
        "description": "Whisper-CD 高斯噪声注入的 SNR 值"
    },
    "whispercd_temporal_shift": {
        "default": 7.0,
        "range": [0.0, 15.0],
        "description": "Whisper-CD 音频时间移位的秒数"
    },
    "whispercd_score_threshold": {
        "default": 0.3,
        "range": [0.0, 1.0],
        "description": "Whisper-CD 一致性分数阈值"
    },
    "whispercd_batch_size": {
        "default": 10,
        "range": [1, 16],
        "description": "Whisper-CD 批量处理大小"
    },
    "whispercd_context_segments": {
        "default": 1,
        "range": [1, 500],
        "description": "Whisper-CD 上下文片段数"
    },
    "enable_forced_alignment": {
        "default": False,
        "description": "是否启用强制对齐（Wav2Vec2/CTC）"
    },
    "translation_batch_size": {
        "default": 2000,
        "range": [1024, 8192],
        "description": "翻译批处理大小（基于token数）"
    },
    "translation_context_size": {
        "default": 8192,
        "range": [1024, 32768],
        "description": "翻译模型上下文大小"
    },
    "translation_temperature": {
        "default": 0.0,
        "range": [0.0, 2.0],
        "description": "翻译模型温度参数"
    },
    "translation_top_k": {
        "default": 20,
        "range": [1, 100],
        "description": "翻译模型Top-K采样参数"
    },
    "translation_top_p": {
        "default": 0.6,
        "range": [0.0, 1.0],
        "description": "翻译模型Top-P（核采样）参数"
    },
    "translation_repetition_penalty": {
        "default": 1.05,
        "range": [1.0, 2.0],
        "description": "翻译模型重复惩罚参数"
    },
    "llama_server_host": {
        "default": "127.0.0.1",
        "description": "Llama Server 服务器地址"
    },
    "llama_server_port": {
        "default": 8080,
        "range": [1, 65535],
        "description": "Llama Server 服务器端口"
    },
    "llama_server_context_size": {
        "default": 8192,
        "range": [512, 32768],
        "description": "Llama Server 上下文大小"
    },
    "llama_server_threads": {
        "default": 8,
        "range": [1, 128],
        "description": "Llama Server 线程数"
    },
    "translation_reset_session": {
        "default": True,
        "description": "是否在每次单句翻译前重置会话状态，确保翻译一致性"
    },
}


def _get_default_config() -> Dict[str, Any]:
    return {key: defn["default"] for key, defn in PARAM_DEFINITIONS.items()}


def _get_type(value: Any) -> type:
    if isinstance(value, bool):
        return bool
    if isinstance(value, int):
        return int
    if isinstance(value, float):
        return float
    return str


class ConfigManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._data = _get_default_config()
            cls._instance._errors = []
            cls._instance._load()
        return cls._instance

    @staticmethod
    def _validate_type(key: str, value: Any, expected_type: type) -> Tuple[bool, str]:
        if value is None:
            return True, ""
        if expected_type == bool:
            return isinstance(value, bool), f"参数 {key} 类型错误: 期望 bool，实际 {type(value).__name__}"
        if expected_type == int:
            if isinstance(value, bool):
                return False, f"参数 {key} 不能是布尔值，应为整数"
            return isinstance(value, int) or (isinstance(value, float) and value.is_integer()), \
                f"参数 {key} 类型错误: 期望 int，实际 {type(value).__name__}"
        if expected_type == float:
            return isinstance(value, (int, float)) and not isinstance(value, bool), \
                f"参数 {key} 类型错误: 期望 float，实际 {type(value).__name__}"
        if expected_type == str:
            return isinstance(value, str), f"参数 {key} 类型错误: 期望 str，实际 {type(value).__name__}"
        return True, ""

    @staticmethod
    def _validate_range(key: str, value: Any, range_def: List[float]) -> Tuple[bool, str]:
        if value is None:
            return True, ""
        min_val, max_val = range_def
        return min_val <= value <= max_val, \
            f"参数 {key} 超出范围: {value} 不在 [{min_val}, {max_val}] 内"

    @staticmethod
    def _validate_options(key: str, value: Any, options: List[Any]) -> Tuple[bool, str]:
        if value is None:
            return True, ""
        return value in options, f"参数 {key} 值无效: {value} 不在可选值 {options} 中"

    @classmethod
    def _validate_param(cls, key: str, value: Any, definition: Dict[str, Any]) -> Tuple[bool, str]:
        expected_type = _get_type(definition["default"])
        valid, msg = cls._validate_type(key, value, expected_type)
        if not valid:
            return False, msg
        if "range" in definition and value is not None:
            valid, msg = cls._validate_range(key, value, definition["range"])
            if not valid:
                return False, msg
        if "options" in definition and value is not None:
            valid, msg = cls._validate_options(key, value, definition["options"])
            if not valid:
                return False, msg
        return True, ""

    def _validate_all(self) -> bool:
        self._errors = []
        for key, value in self._data.items():
            definition = PARAM_DEFINITIONS.get(key)
            if definition:
                valid, msg = self._validate_param(key, value, definition)
                if not valid:
                    self._errors.append(msg)
                    default_value = definition.get("default")
                    if default_value is not None:
                        self._data[key] = default_value
                        print(f"[配置警告] {msg}，已重置为默认值: {default_value}")
        return len(self._errors) == 0

    def _load(self):
        if not os.path.exists(CONFIG_FILE):
            return
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                user_data = json.load(f)
            loaded_count = 0
            for key in self._data:
                if key in user_data:
                    definition = PARAM_DEFINITIONS.get(key)
                    if definition:
                        valid, msg = self._validate_param(key, user_data[key], definition)
                        if valid:
                            self._data[key] = user_data[key]
                            loaded_count += 1
                        else:
                            print(f"[配置警告] {msg}，使用默认值")
            print(f"[配置] 已加载 {loaded_count} 个自定义参数")
            self._validate_all()
        except json.JSONDecodeError as e:
            print(f"[配置错误] 配置文件格式错误: {e}，使用默认配置")
        except Exception as e:
            print(f"[配置错误] 加载失败: {e}，使用默认配置")

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def get_all(self) -> Dict[str, Any]:
        return self._data.copy()

    def set(self, key: str, value: Any) -> Tuple[bool, str]:
        if key not in self._data:
            return False, f"未知配置项: {key}"
        definition = PARAM_DEFINITIONS.get(key)
        if definition:
            valid, msg = self._validate_param(key, value, definition)
            if not valid:
                return False, msg
        self._data[key] = value
        return True, f"已设置 {key} = {value}"

    def save(self, **kwargs) -> Tuple[bool, str]:
        try:
            for key, value in kwargs.items():
                if key in self._data:
                    valid, msg = self.set(key, value)
                    if not valid:
                        return False, msg
            defaults = _get_default_config()
            to_save = {k: v for k, v in self._data.items() if k in defaults}
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(to_save, f, indent=2, ensure_ascii=False)
            return True, f"已保存 {len(to_save)} 个自定义参数"
        except Exception as e:
            return False, f"保存失败: {e}"

    def reset(self) -> Tuple[bool, str]:
        self._data = _get_default_config()
        self._errors = []
        if os.path.exists(CONFIG_FILE):
            try:
                os.remove(CONFIG_FILE)
            except Exception as e:
                return False, f"删除配置文件失败: {e}"
        return True, "已恢复默认配置"

    def build_params(self, **overrides) -> Dict[str, Any]:
        params = self._data.copy()
        for key in _get_default_config():
            if key in overrides:
                params[key] = overrides[key]
        return params

    def get_validation_errors(self) -> List[str]:
        return self._errors.copy()

    def is_valid(self) -> bool:
        return len(self._errors) == 0

    def ui_values(self) -> Dict[str, Any]:
        return {
            "model": self._data.get('model'),
            "device": self._data.get('device'),
            "source_language": self._data.get('source_language'),
            "target_language": self._data.get('target_language'),
            "vad_threshold": self._data.get('vad_threshold'),
            "vad_min_speech_duration": self._data.get('vad_min_speech_duration'),
            "vad_max_speech_duration": self._data.get('vad_max_speech_duration'),
            "vad_min_silence_duration": self._data.get('vad_min_silence_duration'),
            "vad_speech_pad_ms": self._data.get('vad_speech_pad_ms'),
            "enable_whispercd": self._data.get('enable_whispercd'),
            "enable_forced_alignment": self._data.get('enable_forced_alignment'),
            "translation_batch_size": self._data.get('translation_batch_size'),
        }


WHISPER_MODEL_PATTERNS: Dict[str, List[str]] = {
    "tiny": ["tiny", "openai--whisper-tiny"],
    "base": ["base", "openai--whisper-base"],
    "small": ["small", "openai--whisper-small"],
    "medium": ["medium", "openai--whisper-medium"],
    "large-v2": ["large-v2", "openai--whisper-large-v2"],
    "large-v3": ["large-v3", "openai--whisper-large-v3"],
    "large-v3-turbo": ["large-v3-turbo", "openai--whisper-large-v3-turbo"],
}


def get_available_models() -> List[str]:
    available_models = []
    if os.path.exists(MODEL_CACHE_DIR):
        for item in os.listdir(MODEL_CACHE_DIR):
            item_path = os.path.join(MODEL_CACHE_DIR, item)
            if os.path.isdir(item_path):
                item_lower = item.lower()
                matched = None
                for model_name, patterns in WHISPER_MODEL_PATTERNS.items():
                    if any(p in item_lower for p in patterns):
                        if matched is None or len(model_name) > len(matched):
                            matched = model_name
                if matched and matched not in available_models:
                    available_models.append(matched)
    if not available_models:
        available_models = ["tiny", "base", "small", "medium", "large-v2", "large-v3", "large-v3-turbo"]
    model_priority = ["tiny", "base", "small", "medium", "large-v2", "large-v3-turbo", "large-v3"]
    available_models.sort(key=lambda x: model_priority.index(x) if x in model_priority else 999)
    return available_models


MODEL_OPTIONS: List[str] = get_available_models()

LANGUAGE_OPTIONS: List[Tuple[str, str]] = [
    ("auto", "自动检测"),
    ("zh", "中文"),
    ("en", "英语"),
    ("ja", "日语"),
    ("ko", "韩语"),
    ("fr", "法语"),
    ("de", "德语"),
    ("es", "西班牙语"),
    ("ru", "俄语")
]

DEVICE_OPTIONS: List[str] = ["auto", "cuda", "cpu"]

TRANSLATOR_QUANTIZATION_OPTIONS: List[str] = ["auto", "Q4_K_M", "Q6_K", "Q8_0"]

config = ConfigManager()

__all__ = [
    'ConfigManager',
    'config',
    'PARAM_DEFINITIONS',
    'MODEL_OPTIONS',
    'LANGUAGE_OPTIONS',
    'DEVICE_OPTIONS',
    'TRANSLATOR_QUANTIZATION_OPTIONS',
    'PROJECT_ROOT',
    'MODEL_CACHE_DIR',
    'TEMP_DIR',
    'OUTPUT_DIR',
    'CONFIG_FILE'
]
