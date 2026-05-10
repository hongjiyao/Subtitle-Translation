# -*- coding: utf-8 -*-
"""
配置管理模块
提供系统化的配置参数定义、验证和管理功能
"""

import os
import json
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, fields


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_CACHE_DIR = os.path.join(PROJECT_ROOT, "models")
TEMP_DIR = os.path.join(PROJECT_ROOT, "temp")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
CONFIG_FILE = os.path.join(PROJECT_ROOT, "saved_params.json")

for d in [MODEL_CACHE_DIR, TEMP_DIR, OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)

os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
os.environ.setdefault('LC_ALL', 'en_US.UTF-8')
os.environ.setdefault('LANG', 'en_US.UTF-8')
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HUGGINGFACE_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HOME', str(MODEL_CACHE_DIR))
os.environ.setdefault('HUGGINGFACE_HUB_CACHE', MODEL_CACHE_DIR)


@dataclass
class VadParams:
    threshold: float = 0.5
    min_speech: float = 0.5
    max_speech: float = 30.0
    min_silence: float = 0.5
    speech_pad_ms: int = 400
    prefix_padding_ms: int = 30
    neg_threshold: float = 0.1
    max_sil_split: bool = True
    max_sil_dur: int = 98
    resolution: int = 1

    _KEY_MAP = {
        'threshold': 'vad_threshold',
        'min_speech': 'vad_min_speech_duration',
        'max_speech': 'vad_max_speech_duration',
        'min_silence': 'vad_min_silence_duration',
        'speech_pad_ms': 'vad_speech_pad_ms',
        'prefix_padding_ms': 'vad_prefix_padding_ms',
        'neg_threshold': 'vad_neg_threshold',
        'max_sil_split': 'use_max_poss_sil_at_max_speech',
        'max_sil_dur': 'vad_min_silence_at_max_speech',
        'resolution': 'vad_time_resolution',
    }

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> 'VadParams':
        kwargs = {}
        for f in fields(cls):
            if f.name.startswith('_'):
                continue
            key = cls._KEY_MAP.get(f.name, f.name)
            if key in params:
                kwargs[f.name] = params[key]
        return cls(**kwargs)


@dataclass
class CdParams:
    alpha: float = 1.0
    temperature: float = 1.0
    snr_db: float = 10.0
    temporal_shift: float = 7.0
    ctx_tokens: int = 200

    _KEY_MAP = {
        'alpha': 'whispercd_alpha',
        'temperature': 'whispercd_temperature',
        'snr_db': 'whispercd_snr_db',
        'temporal_shift': 'whispercd_temporal_shift',
        'ctx_tokens': 'whispercd_context_max_tokens',
    }

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> 'CdParams':
        kwargs = {}
        for f in fields(cls):
            if f.name.startswith('_'):
                continue
            key = cls._KEY_MAP.get(f.name, f.name)
            if key in params:
                kwargs[f.name] = params[key]
        return cls(**kwargs)


@dataclass
class TransParams:
    temperature: float = 0.2
    top_k: int = 20
    top_p: float = 0.6
    rep_penalty: float = 1.05
    batch_size: int = 2000
    ctx_size: int = 8192
    seg_ctx_window: int = 6
    max_retries: int = 3
    max_total_retries: int = 10
    max_output_tokens: int = 8000
    reset_session: bool = True

    _KEY_MAP = {
        'temperature': 'translation_temperature',
        'top_k': 'translation_top_k',
        'top_p': 'translation_top_p',
        'rep_penalty': 'translation_repetition_penalty',
        'batch_size': 'translation_batch_size',
        'ctx_size': 'translation_context_size',
        'seg_ctx_window': 'translation_segment_context_window',
        'max_retries': 'translation_max_retries',
        'max_total_retries': 'translation_max_total_retries',
        'max_output_tokens': 'translation_max_output_tokens',
        'reset_session': 'translation_reset_session',
    }

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> 'TransParams':
        kwargs = {}
        for f in fields(cls):
            if f.name.startswith('_'):
                continue
            key = cls._KEY_MAP.get(f.name, f.name)
            if key in params:
                kwargs[f.name] = params[key]
        return cls(**kwargs)


@dataclass
class ServerParams:
    host: str = "127.0.0.1"
    port: int = 8080
    ctx_size: int = 8192
    threads: int = 8
    ngl: int = 99
    batch_size: int = 2048

    _KEY_MAP = {
        'host': 'llama_server_host',
        'port': 'llama_server_port',
        'ctx_size': 'llama_server_context_size',
        'threads': 'llama_server_threads',
        'ngl': 'llama_server_ngl',
        'batch_size': 'llama_server_batch_size',
    }

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> 'ServerParams':
        kwargs = {}
        for f in fields(cls):
            if f.name.startswith('_'):
                continue
            key = cls._KEY_MAP.get(f.name, f.name)
            if key in params:
                kwargs[f.name] = params[key]
        return cls(**kwargs)


PARAM_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "model": {
        "default": "large-v3-turbo",
        "options": ["tiny", "base", "small", "medium", "large-v2", "large-v3", "large-v3-turbo"],
        "description": "Whisper 语音识别模型名称，影响识别精度和速度"
    },
    "translator": {
        "default": "tencent/HY-MT1.5-7B-GGUF",
        "options": ["tencent/HY-MT1.5-7B-GGUF"],
        "description": "翻译模型标识，目前仅支持 tencent/HY-MT1.5-7B-GGUF"
    },
    "translator_quantization": {
        "default": "Q4_K_M",
        "options": ["auto", "Q4_K_M", "Q6_K", "Q8_0"],
        "description": "翻译模型量化版本，auto 自动选择最小可用模型"
    },
    "device": {
        "default": "auto",
        "options": ["auto", "cuda", "cpu"],
        "description": "计算设备选择，auto 自动检测 CUDA 可用性"
    },
    "source_language": {
        "default": "ja",
        "options": ["zh", "en", "ja", "ko", "fr", "de", "es", "ru", "ar", "hi", "pt", "it", "nl", "pl"],
        "description": "源语言（音频输入语言），用于语音识别和翻译方向"
    },
    "target_language": {
        "default": "zh",
        "options": ["zh", "en", "ja", "ko", "fr", "de", "es", "ru", "ar", "hi", "pt", "it", "nl", "pl"],
        "description": "目标语言（翻译输出语言）"
    },
    "vad_threshold": {
        "default": 0.02,
        "range": [0.0, 1.0],
        "description": "VAD 语音活动检测概率阈值，值越高要求越确定才判定为语音"
    },
    "vad_min_speech_duration": {
        "default": 0.5,
        "range": [0.01, 10.0],
        "description": "VAD 最短语音持续时间（秒），短于此值的片段被过滤"
    },
    "vad_max_speech_duration": {
        "default": 30.0,
        "range": [1.0, 300.0],
        "description": "VAD 最长语音持续时间（秒），超长片段会被强制分割"
    },
    "vad_min_silence_duration": {
        "default": 0.2,
        "range": [0.01, 10.0],
        "description": "VAD 最短静默持续时间（秒），用于分割相邻语音片段"
    },
    "vad_speech_pad_ms": {
        "default": 0,
        "range": [0, 5000],
        "description": "VAD 语音片段前后填充时间（毫秒），避免边缘截断"
    },
    "vad_prefix_padding_ms": {
        "default": 300,
        "range": [0, 5000],
        "description": "VAD 前缀填充时间（毫秒），在每个语音片段开头添加额外音频"
    },
    "vad_neg_threshold": {
        "default": 0.01,
        "range": [0.0, 1.0],
        "description": "VAD 静音判定阈值，低于此值判定为静音，用于语音片段结束检测"
    },
    "use_max_poss_sil_at_max_speech": {
        "default": True,
        "description": "VAD 在最长语音片段中是否优先使用最长静音点进行分割"
    },
    "vad_min_silence_at_max_speech": {
        "default": 98,
        "range": [0, 1000],
        "description": "VAD 长语音分割时最大片段内静音时长（毫秒），用于避免强制分割时的突兀截断"
    },
    "vad_time_resolution": {
        "default": 1,
        "range": [1, 1000],
        "description": "VAD 时间分辨率（毫秒），控制语音时间戳的精度"
    },
    "whispercd_alpha": {
        "default": 1.0,
        "range": [0.0, 2.0],
        "description": "Whisper-CD 对比强度参数，值越大对比解码影响越强"
    },
    "whispercd_temperature": {
        "default": 1.0,
        "range": [0.1, 5.0],
        "description": "Whisper-CD log-sum-exp 温度参数，控制扰动 logits 的聚合平滑度"
    },
    "whispercd_snr_db": {
        "default": 10.0,
        "range": [0.0, 30.0],
        "description": "Whisper-CD 高斯噪声注入的信噪比（dB），值越高噪声越小"
    },
    "whispercd_temporal_shift": {
        "default": 7.0,
        "range": [0.0, 15.0],
        "description": "Whisper-CD 音频时间移位的秒数，用于生成时间扰动负样本"
    },
    "whispercd_context_max_tokens": {
        "default": 200,
        "range": [64, 400],
        "description": "Whisper-CD 上下文最大字符数（实际 token 数可能更多，超限时自动截断至 token 预算内）"
    },
    "enable_forced_alignment": {
        "default": True,
        "description": "是否启用 Wav2Vec2 强制对齐，获取字符级和词级精确时间戳"
    },
    "translation_batch_size": {
        "default": 2000,
        "range": [1024, 8192],
        "description": "翻译批处理大小（基于 token 数），影响单次翻译的文本量"
    },
    "translation_context_size": {
        "default": 4096,
        "range": [1024, 32768],
        "description": "翻译模型上下文窗口大小，限制单次翻译可用的总 token 数"
    },
    "translation_segment_context_window": {
        "default": 6,
        "range": [0, 20],
        "description": "翻译时读取前后片段的数量，用于提供上下文信息"
    },
    "translation_temperature": {
        "default": 0.2,
        "range": [0.0, 2.0],
        "description": "翻译模型采样温度，值越低输出越确定，值越高越多样"
    },
    "translation_top_k": {
        "default": 20,
        "range": [1, 100],
        "description": "翻译模型 Top-K 采样参数，限制候选 token 数量"
    },
    "translation_top_p": {
        "default": 0.6,
        "range": [0.0, 1.0],
        "description": "翻译模型 Top-P 核采样参数，限制累积概率阈值"
    },
    "translation_repetition_penalty": {
        "default": 1.05,
        "range": [1.0, 2.0],
        "description": "翻译模型重复惩罚参数，值大于 1 抑制重复生成"
    },
    "llama_server_host": {
        "default": "127.0.0.1",
        "description": "llama-server 监听地址，通常为 127.0.0.1"
    },
    "llama_server_port": {
        "default": 8080,
        "range": [1, 65535],
        "description": "llama-server 监听端口"
    },
    "llama_server_context_size": {
        "default": 8192,
        "range": [512, 32768],
        "description": "llama-server 上下文窗口大小，限制单次请求的 prompt+输出总 token 数"
    },
    "llama_server_threads": {
        "default": 8,
        "range": [1, 128],
        "description": "llama-server CPU 线程数"
    },
    "translation_reset_session": {
        "default": True,
        "description": "是否在每次翻译前重置 llama-server 会话，确保翻译一致性"
    },
    "llama_server_ngl": {
        "default": 99,
        "range": [0, 999],
        "description": "llama-server GPU 卸载层数，0 表示仅 CPU，99 表示全部卸载到 GPU"
    },
    "llama_server_batch_size": {
        "default": 2048,
        "range": [512, 8192],
        "description": "llama-server 批处理大小，影响推理速度和显存占用"
    },
    "translation_max_retries": {
        "default": 3,
        "range": [1, 10],
        "description": "翻译单条最大重试次数，超过后使用原文"
    },
    "translation_max_total_retries": {
        "default": 10,
        "range": [3, 30],
        "description": "翻译验证失败最大总重试次数，超过后使用原文"
    },
    "translation_max_output_tokens": {
        "default": 8000,
        "range": [1024, 32768],
        "description": "翻译最大输出 token 数，限制单次翻译输出长度"
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
            "vad_prefix_padding_ms": self._data.get('vad_prefix_padding_ms'),
            "vad_neg_threshold": self._data.get('vad_neg_threshold'),
            "use_max_poss_sil_at_max_speech": self._data.get('use_max_poss_sil_at_max_speech'),
            "vad_min_silence_at_max_speech": self._data.get('vad_min_silence_at_max_speech'),
            "vad_time_resolution": self._data.get('vad_time_resolution'),
            "enable_forced_alignment": self._data.get('enable_forced_alignment'),
            "translation_batch_size": self._data.get('translation_batch_size'),
            "translation_context_size": self._data.get('translation_context_size'),
            "translation_segment_context_window": self._data.get('translation_segment_context_window'),
            "translation_temperature": self._data.get('translation_temperature'),
            "translation_top_k": self._data.get('translation_top_k'),
            "translation_top_p": self._data.get('translation_top_p'),
            "translation_repetition_penalty": self._data.get('translation_repetition_penalty'),
            "translation_max_retries": self._data.get('translation_max_retries'),
            "translation_max_total_retries": self._data.get('translation_max_total_retries'),
            "translation_max_output_tokens": self._data.get('translation_max_output_tokens'),
            "llama_server_host": self._data.get('llama_server_host'),
            "llama_server_port": self._data.get('llama_server_port'),
            "llama_server_context_size": self._data.get('llama_server_context_size'),
            "llama_server_threads": self._data.get('llama_server_threads'),
            "llama_server_ngl": self._data.get('llama_server_ngl'),
            "llama_server_batch_size": self._data.get('llama_server_batch_size'),
            "whispercd_alpha": self._data.get('whispercd_alpha'),
            "whispercd_temperature": self._data.get('whispercd_temperature'),
            "whispercd_snr_db": self._data.get('whispercd_snr_db'),
            "whispercd_temporal_shift": self._data.get('whispercd_temporal_shift'),
            "whispercd_context_max_tokens": self._data.get('whispercd_context_max_tokens'),
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
    'CONFIG_FILE',
    'VadParams',
    'CdParams',
    'TransParams',
    'ServerParams',
]
