# -*- coding: utf-8 -*-
"""
配置管理模块
提供系统化的配置参数定义、验证和管理功能
"""

import os
import re
import json
import threading
from typing import Dict, Any, List, Tuple, Optional, ClassVar
from dataclasses import dataclass, fields

def _detect_package_mode():
    _current_dir = os.path.dirname(os.path.abspath(__file__))
    _parent_dir = os.path.dirname(_current_dir)
    return os.path.isfile(os.path.join(_parent_dir, "python", "python.exe"))

IS_PACKAGE_MODE = _detect_package_mode()

if IS_PACKAGE_MODE:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _python_dir = os.path.join(PROJECT_ROOT, "python")
    if hasattr(os, 'add_dll_directory'):
        os.add_dll_directory(_python_dir)
else:
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
class ParamBase:
    _KEY_MAP: ClassVar[Dict[str, str]] = {}

    @classmethod
    def from_dict(cls, params):
        kwargs = {}
        for f in fields(cls):
            if f.name.startswith('_'):
                continue
            key = cls._KEY_MAP.get(f.name, f.name)
            if key in params:
                kwargs[f.name] = params[key]
        return cls(**kwargs)



@dataclass
class CdParams(ParamBase):
    alpha: float = 0.5                    # Whisper-CD 对比强度参数，值越大对比解码影响越强，范围 [0.0, 2.0]
    temperature: float = 1.0              # Whisper-CD log-sum-exp 温度参数，控制扰动 logits 的聚合平滑度，范围 [0.1, 5.0]
    snr_db: float = 10.0                  # Whisper-CD 高斯噪声注入的信噪比（dB），值越高噪声越小，范围 [0.0, 30.0]
    temporal_shift: float = 7.0           # Whisper-CD 音频时间移位的秒数，用于生成时间扰动负样本，范围 [0.0, 15.0]
    ctx_tokens: int = 200                 # Whisper-CD 上下文最大字符数，范围 [64, 400]
    max_duration: float = 3.0             # 字幕最大允许时长（秒），超过必须分割，范围 [2.0, 15.0]
    min_duration: float = 1.5             # 字幕碎片阈值（秒），低于此值视为碎片考虑合并，范围 [0.5, 5.0]
    merge_max_duration: float = 4.0       # 合并安全上限（秒），合并后绝不超过此值，范围 [1.0, 15.0]
    gap_threshold: float = 2.0           # 跨chunk边界合并的最大间隔（秒），间隔小于此值且语义连续时合并，范围 [0.5, 5.0]
    particle_chars: str = "てにをはがのともへでかよねば"  # 三级分割自然断点字符，用于无标点时长片段的分割
    silence_threshold: float = 0.95        # 静音检测余弦相似度阈值，高于此值判定为静音跳过，范围 [0.8, 1.0]
    max_token_repeat: int = 3              # 单token重复抑制阈值，同一token出现次数≥此值时抑制，范围 [2, 10]
    long_seq_window: int = 20              # 长序列重复检测滑动窗口大小，范围 [5, 50]
    long_seq_threshold: float = 0.7        # 长序列重复相似度阈值，高于此值判定为重复，范围 [0.3, 1.0]
    target_token_count: int = 25           # 助词级分割目标token数，范围 [10, 50]
    search_range: int = 6                  # 助词级分割搜索范围，在目标位置前后搜索的token数，范围 [2, 15]
    low_confidence_threshold: float = -0.5 # 低置信度转录警告阈值，avg_logprob低于此值触发警告，范围 [-2.0, 0.0]
    continuation_gap_multiplier: float = 1.5  # 续行时放宽间隔阈值的倍数，范围 [1.0, 3.0]

    _KEY_MAP: ClassVar[Dict[str, str]] = {
        'alpha': 'whispercd_alpha',
        'temperature': 'whispercd_temperature',
        'snr_db': 'whispercd_snr_db',
        'temporal_shift': 'whispercd_temporal_shift',
        'ctx_tokens': 'whispercd_context_max_tokens',
        'max_duration': 'whispercd_max_duration',
        'min_duration': 'whispercd_min_duration',
        'merge_max_duration': 'whispercd_merge_max_duration',
        'gap_threshold': 'whispercd_gap_threshold',
        'particle_chars': 'whispercd_particle_chars',
        'silence_threshold': 'whispercd_silence_threshold',
        'max_token_repeat': 'whispercd_max_token_repeat',
        'long_seq_window': 'whispercd_long_seq_window',
        'long_seq_threshold': 'whispercd_long_seq_threshold',
        'target_token_count': 'whispercd_target_token_count',
        'search_range': 'whispercd_search_range',
        'low_confidence_threshold': 'whispercd_low_confidence_threshold',
        'continuation_gap_multiplier': 'whispercd_continuation_gap_multiplier',
    }


@dataclass
class TransParams(ParamBase):
    temperature: float = 0.1               # 翻译模型采样温度，值越低输出越确定，值越高越多样，范围 [0.0, 2.0]
    top_k: int = 20                        # 翻译模型 Top-K 采样参数，限制候选 token 数量，范围 [1, 100]
    top_p: float = 0.6                     # 翻译模型 Top-P 核采样参数，限制累积概率阈值，范围 [0.0, 1.0]
    rep_penalty: float = 1.05              # 翻译模型重复惩罚参数，值大于 1 抑制重复生成，范围 [1.0, 2.0]
    seg_ctx_window: int = 3                # 翻译时读取前后片段的数量，用于提供上下文信息，范围 [0, 20]
    max_context_tokens: int = 512          # 翻译上下文片段的最大 token 数，超出时自动截断，范围 [0, 4096]
    max_retries: int = 3                   # 翻译单条最大重试次数，超过后使用原文，范围 [1, 10]
    max_total_retries: int = 3            # 翻译验证失败最大总重试次数，超过后使用原文，范围 [3, 30]
    max_output_tokens: int = 512           # 翻译最大输出 token 数，限制单次翻译输出长度，范围 [64, 4096]
    reset_session: bool = False            # 是否在每次翻译前重置 llama-server 会话，确保翻译一致性
    validation_threshold: float = 0.5      # 翻译验证目标语言占比阈值，低于此值判定为翻译失败，范围 [0.1, 1.0]
    short_text_threshold: int = 5          # 短文本判断阈值（字符数），低于此值视为短文本特殊处理，范围 [1, 20]
    kana_ratio_threshold: float = 0.3      # 日译中时假名占比阈值，高于此值判定为未翻译，范围 [0.1, 0.8]
    request_timeout: int = 300             # 翻译HTTP请求超时时间（秒），范围 [30, 600]

    _KEY_MAP: ClassVar[Dict[str, str]] = {
        'temperature': 'translation_temperature',
        'top_k': 'translation_top_k',
        'top_p': 'translation_top_p',
        'rep_penalty': 'translation_repetition_penalty',
        'seg_ctx_window': 'translation_segment_context_window',
        'max_context_tokens': 'translation_max_context_tokens',
        'max_retries': 'translation_max_retries',
        'max_total_retries': 'translation_max_total_retries',
        'max_output_tokens': 'translation_max_output_tokens',
        'reset_session': 'translation_reset_session',
        'validation_threshold': 'translation_validation_threshold',
        'short_text_threshold': 'translation_short_text_threshold',
        'kana_ratio_threshold': 'translation_kana_ratio_threshold',
        'request_timeout': 'translation_request_timeout',
    }


@dataclass
class ServerParams(ParamBase):
    host: str = "127.0.0.1"                # llama-server 监听地址，通常为 127.0.0.1
    port: int = 8080                        # llama-server 监听端口，范围 [1, 65535]
    ctx_size: int = 4096                    # llama-server 上下文窗口大小，限制单次请求的 prompt+输出总 token 数，范围 [512, 32768]
    threads: int = 8                        # llama-server CPU 线程数，范围 [1, 128]
    ngl: int = 99                           # llama-server GPU 卸载层数，0 表示仅 CPU，99 表示全部卸载到 GPU，范围 [0, 999]
    batch_size: int = 512                  # llama-server 批处理大小，影响推理速度和显存占用，范围 [512, 8192]
    parallel_slots: int = 1                 # llama-server 并行处理槽数，单用户建议设为1以释放VRAM提升速度，范围 [1, 8]

    _KEY_MAP: ClassVar[Dict[str, str]] = {
        'host': 'llama_server_host',
        'port': 'llama_server_port',
        'ctx_size': 'llama_server_context_size',
        'threads': 'llama_server_threads',
        'ngl': 'llama_server_ngl',
        'batch_size': 'llama_server_batch_size',
        'parallel_slots': 'llama_server_parallel_slots',
    }


def _build_param_definitions() -> Dict[str, Dict[str, Any]]:
    manual_params = {
        "model": {
            "default": "large-v3-turbo",
            "options": ["tiny", "base", "small", "medium", "large-v2", "large-v3", "large-v3-turbo"],
            "description": "Whisper 语音识别模型名称，影响识别精度和速度"
        },
        "translator": {
            "default": "tencent/HY-MT1.5-1.8B-GGUF-Q8_0",
            "options": ["tencent/HY-MT1.5-1.8B-GGUF-Q8_0", "SakuraLLM/Sakura-7B-Qwen2.5-v1.0-GGUF-Q6_K"],
            "description": "翻译模型标识（含量化版本），支持 tencent/HY-MT1.5-1.8B-GGUF-Q8_0 和 SakuraLLM/Sakura-7B-Qwen2.5-v1.0-GGUF-Q6_K"
        },
        "device": {
            "default": "cuda",
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
        "enable_forced_alignment": {
            "default": True,
            "description": "是否启用 Wav2Vec2 强制对齐，获取字符级和词级精确时间戳"
        },
    }

    param_metadata = {
        "whispercd_alpha": {"range": [0.0, 2.0], "description": "Whisper-CD 对比强度参数，值越大对比解码影响越强"},
        "whispercd_temperature": {"range": [0.1, 5.0], "description": "Whisper-CD log-sum-exp 温度参数，控制扰动 logits 的聚合平滑度"},
        "whispercd_snr_db": {"range": [0.0, 30.0], "description": "Whisper-CD 高斯噪声注入的信噪比（dB），值越高噪声越小"},
        "whispercd_temporal_shift": {"range": [0.0, 15.0], "description": "Whisper-CD 音频时间移位的秒数，用于生成时间扰动负样本"},
        "whispercd_context_max_tokens": {"range": [64, 400], "description": "Whisper-CD 上下文最大字符数（实际 token 数可能更多，超限时自动截断至 token 预算内）"},
        "whispercd_max_duration": {"range": [2.0, 15.0], "description": "字幕最大允许时长（秒），超过必须分割"},
        "whispercd_min_duration": {"range": [0.5, 5.0], "description": "字幕碎片阈值（秒），低于此值视为碎片考虑合并"},
        "whispercd_merge_max_duration": {"range": [1.0, 15.0], "description": "合并安全上限（秒），合并后绝不超过此值"},
        "whispercd_gap_threshold": {"range": [0.5, 5.0], "description": "跨chunk边界合并的最大间隔（秒），间隔小于此值且语义连续时合并"},
        "whispercd_particle_chars": {"description": "三级分割自然断点字符，用于无标点时长片段的分割。日语助词如：てにをはがのともへでかよねば；中文可填：的了是在不和有这"},
        "whispercd_silence_threshold": {"range": [0.8, 1.0], "description": "静音检测余弦相似度阈值，高于此值判定为静音跳过，值越高越不易误判"},
        "whispercd_max_token_repeat": {"range": [2, 10], "description": "单token重复抑制阈值，同一token出现次数≥此值时抑制"},
        "whispercd_long_seq_window": {"range": [5, 50], "description": "长序列重复检测滑动窗口大小"},
        "whispercd_long_seq_threshold": {"range": [0.3, 1.0], "description": "长序列重复相似度阈值，高于此值判定为重复"},
        "whispercd_target_token_count": {"range": [10, 50], "description": "助词级分割目标token数，控制分割后每段的大致token数量"},
        "whispercd_search_range": {"range": [2, 15], "description": "助词级分割搜索范围，在目标位置前后搜索的token数"},
        "whispercd_low_confidence_threshold": {"range": [-2.0, 0.0], "description": "低置信度转录警告阈值，avg_logprob低于此值触发警告"},
        "whispercd_continuation_gap_multiplier": {"range": [1.0, 3.0], "description": "续行时放宽间隔阈值的倍数，检测到续行助词时gap_threshold乘以此值"},
        "translation_temperature": {"range": [0.0, 2.0], "description": "翻译模型采样温度，值越低输出越确定，值越高越多样"},
        "translation_top_k": {"range": [1, 100], "description": "翻译模型 Top-K 采样参数，限制候选 token 数量"},
        "translation_top_p": {"range": [0.0, 1.0], "description": "翻译模型 Top-P 核采样参数，限制累积概率阈值"},
        "translation_repetition_penalty": {"range": [1.0, 2.0], "description": "翻译模型重复惩罚参数，值大于 1 抑制重复生成"},
        "translation_segment_context_window": {"range": [0, 20], "description": "翻译时读取前后片段的数量，用于提供上下文信息"},
        "translation_max_context_tokens": {"range": [0, 4096], "description": "翻译上下文片段的最大 token 数，超出时自动截断最旧的上下文片段，设为0则不使用上下文"},
        "translation_max_retries": {"range": [1, 10], "description": "翻译单条最大重试次数，超过后使用原文"},
        "translation_max_total_retries": {"range": [3, 30], "description": "翻译验证失败最大总重试次数，超过后使用原文"},
        "translation_max_output_tokens": {"range": [64, 4096], "description": "翻译最大输出 token 数，限制单次翻译输出长度"},
        "translation_reset_session": {"description": "是否在每次翻译前重置 llama-server 会话，确保翻译一致性"},
        "translation_validation_threshold": {"range": [0.1, 1.0], "description": "翻译验证目标语言占比阈值，低于此值判定为翻译失败"},
        "translation_short_text_threshold": {"range": [1, 20], "description": "短文本判断阈值（字符数），低于此值视为短文本特殊处理"},
        "translation_kana_ratio_threshold": {"range": [0.1, 0.8], "description": "日译中时假名占比阈值，高于此值判定为未翻译"},
        "translation_request_timeout": {"range": [30, 600], "description": "翻译HTTP请求超时时间（秒）"},
        "llama_server_host": {"description": "llama-server 监听地址，通常为 127.0.0.1"},
        "llama_server_port": {"range": [1, 65535], "description": "llama-server 监听端口"},
        "llama_server_context_size": {"range": [512, 32768], "description": "llama-server 上下文窗口大小，限制单次请求的 prompt+输出总 token 数"},
        "llama_server_threads": {"range": [1, 128], "description": "llama-server CPU 线程数"},
        "llama_server_ngl": {"range": [0, 999], "description": "llama-server GPU 卸载层数，0 表示仅 CPU，99 表示全部卸载到 GPU"},
        "llama_server_batch_size": {"range": [512, 8192], "description": "llama-server 批处理大小，影响推理速度和显存占用"},
        "llama_server_parallel_slots": {"range": [1, 8], "description": "llama-server 并行处理槽数，单用户建议设为1以释放VRAM提升速度"},
    }

    result = {}
    for key, defn in manual_params.items():
        result[key] = defn

    dataclass_configs = [
        (CdParams, CdParams._KEY_MAP),
        (TransParams, TransParams._KEY_MAP),
        (ServerParams, ServerParams._KEY_MAP),
    ]

    for cls, key_map in dataclass_configs:
        for field in fields(cls):
            if field.name.startswith('_'):
                continue
            config_key = key_map.get(field.name, field.name)
            entry = {"default": field.default}
            if config_key in param_metadata:
                entry.update(param_metadata[config_key])
            result[config_key] = entry

    return result


PARAM_DEFINITIONS: Dict[str, Dict[str, Any]] = _build_param_definitions()


def _get_default_config() -> Dict[str, Any]:
    return {key: defn["default"] for key, defn in PARAM_DEFINITIONS.items()}


def _get_type(value: Any) -> type:
    if isinstance(value, bool):
        return bool
    if isinstance(value, int):
        return int
    if isinstance(value, float):
        return float
    if value is None:
        return type(None)
    return str


class ConfigManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._data = _get_default_config()
                    cls._instance._errors = []
                    cls._instance._data_lock = threading.Lock()
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

    def _validate_dependencies(self) -> Tuple[bool, str]:
        ctx_size = self._data.get("llama_server_context_size", 0)
        trans_output = self._data.get("translation_max_output_tokens", 0)
        batch_size = self._data.get("llama_server_batch_size", 0)

        if trans_output > ctx_size:
            return False, f"translation_max_output_tokens({trans_output}) 不能大于 llama_server_context_size({ctx_size})"
        if batch_size > ctx_size:
            return False, f"llama_server_batch_size({batch_size}) 不能大于 llama_server_context_size({ctx_size})"
        return True, ""

    def _load(self):
        if not os.path.exists(CONFIG_FILE):
            return
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                user_data = json.load(f)
            loaded_count = 0
            cleaned = False
            for key in user_data:
                if key not in PARAM_DEFINITIONS:
                    print(f"配置文件中存在未知参数: {key}，已自动清理")
                    cleaned = True
                    continue
                if key in self._data:
                    definition = PARAM_DEFINITIONS.get(key)
                    if definition:
                        valid, msg = self._validate_param(key, user_data[key], definition)
                        if valid:
                            self._data[key] = user_data[key]
                            loaded_count += 1
                        else:
                            print(f"[配置警告] {msg}，使用默认值")
            if cleaned:
                self.save()
                print("[配置] 已自动清理配置文件中的未知参数")
            print(f"[配置] 已加载 {loaded_count} 个自定义参数")
            self._validate_all()
        except json.JSONDecodeError as e:
            print(f"[配置错误] 配置文件格式错误: {e}，使用默认配置")
        except Exception as e:
            print(f"[配置错误] 加载失败: {e}，使用默认配置")

    def get(self, key: str, default: Any = None) -> Any:
        with self._data_lock:
            return self._data.get(key, default)

    def get_all(self) -> Dict[str, Any]:
        with self._data_lock:
            return self._data.copy()

    def set(self, key: str, value: Any) -> Tuple[bool, str]:
        if key not in self._data:
            return False, f"未知配置项: {key}"
        if key == "llama_server_host":
            if not isinstance(value, str) or not re.match(r'^[\w.\-]+$', value):
                return False, "无效的主机地址"
        definition = PARAM_DEFINITIONS.get(key)
        if definition:
            valid, msg = self._validate_param(key, value, definition)
            if not valid:
                return False, msg
            expected_type = _get_type(definition["default"])
            if expected_type == int and isinstance(value, float) and value == int(value):
                value = int(value)
            elif expected_type == float and isinstance(value, int):
                value = float(value)
        with self._data_lock:
            self._data[key] = value
        return True, f"已设置 {key} = {value}"

    def save(self, **kwargs) -> Tuple[bool, str]:
        try:
            for key, value in kwargs.items():
                if key in self._data:
                    valid, msg = self.set(key, value)
                    if not valid:
                        return False, msg
            valid, msg = self._validate_dependencies()
            if not valid:
                return False, msg
            defaults = _get_default_config()
            with self._data_lock:
                to_save = {k: v for k, v in self._data.items() if k in defaults}
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(to_save, f, indent=2, ensure_ascii=False)
            return True, f"已保存 {len(to_save)} 个自定义参数"
        except Exception as e:
            return False, f"保存失败: {e}"

    def reset(self) -> Tuple[bool, str]:
        with self._data_lock:
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

TRANSLATOR_OPTIONS: List[str] = ["tencent/HY-MT1.5-1.8B-GGUF-Q8_0", "SakuraLLM/Sakura-7B-Qwen2.5-v1.0-GGUF-Q6_K"]

config = ConfigManager()

__all__ = [
    'ConfigManager',
    'config',
    'PARAM_DEFINITIONS',
    'MODEL_OPTIONS',
    'LANGUAGE_OPTIONS',
    'DEVICE_OPTIONS',
    'TRANSLATOR_OPTIONS',
    'PROJECT_ROOT',
    'MODEL_CACHE_DIR',
    'TEMP_DIR',
    'OUTPUT_DIR',
    'CONFIG_FILE',
    'IS_PACKAGE_MODE',
    'CdParams',
    'TransParams',
    'ServerParams',
]
