# -*- coding: utf-8 -*-
"""
配置管理模块
提供系统化的配置参数定义、验证和管理功能
"""

import os
import json
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field, asdict


# =============================================================================
# 路径配置
# =============================================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_CACHE_DIR = os.path.join(PROJECT_ROOT, "models")
TEMP_DIR = os.path.join(PROJECT_ROOT, "temp")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
CONFIG_FILE = os.path.join(PROJECT_ROOT, "saved_params.json")

# 确保目录存在
for d in [MODEL_CACHE_DIR, TEMP_DIR, OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)


# =============================================================================
# 配置参数定义
# =============================================================================

@dataclass
class ConfigDefinitions:
    """
    配置参数定义类
    所有可配置参数在此定义，包含类型、默认值、约束和说明
    """
    
    # =========================================================================
    # 1. 模型与设备配置 (Model & Device Settings)
    # =========================================================================
    
    # -------------------------------------------------------------------------
    # 模型配置 (Model Settings)
    # -------------------------------------------------------------------------
    model: str = field(
        default="medium",
        metadata={
            "type": str,
            "description": "Whisper/WhisperX 语音识别模型",
            "options": ["tiny", "base", "small", "medium", "large-v2", "large-v3"],
            "constraints": "必须是预定义的模型名称之一"
        }
    )
    
    translator: str = field(
        default="tencent/HY-MT1.5-7B-GGUF",
        metadata={
            "type": str,
            "description": "翻译模型标识",
            "options": ["tencent/HY-MT1.5-7B-GGUF"],
            "constraints": "必须是支持的翻译模型"
        }
    )
    
    translator_quantization: str = field(
        default="Q4_K_M",
        metadata={
            "type": str,
            "description": "翻译模型量化版本选择",
            "options": ["auto", "Q4_K_M", "Q6_K", "Q8_0"],
            "constraints": "auto表示自动选择最小模型，其他选项指定具体量化版本"
        }
    )
    
    device: str = field(
        default="auto",
        metadata={
            "type": str,
            "description": "计算设备选择",
            "options": ["auto", "cuda", "cpu"],
            "constraints": "auto会自动检测GPU可用性"
        }
    )
    
    source_language: str = field(
        default="auto",
        metadata={
            "type": str,
            "description": "源语言（语音输入语言）",
            "options": ["auto", "zh", "en", "ja", "ko", "fr", "de", "es", "ru", "ar", "hi", "pt", "it", "nl", "pl"],
            "constraints": "auto表示自动检测"
        }
    )
    
    target_language: str = field(
        default="zh",
        metadata={
            "type": str,
            "description": "目标语言（翻译输出语言）",
            "options": ["zh", "en", "ja", "ko", "fr", "de", "es", "ru", "ar", "hi", "pt", "it", "nl", "pl"],
            "constraints": "当前主要支持中文翻译"
        }
    )
    
    # -------------------------------------------------------------------------
    # 语音识别配置 (Speech Recognition Settings)
    # -------------------------------------------------------------------------
    vad_filter: bool = field(
        default=True,
        metadata={
            "type": bool,
            "description": "是否启用VAD语音活动检测过滤",
            "constraints": "启用可提高识别精度"
        }
    )
    
    word_timestamps: bool = field(
        default=False,
        metadata={
            "type": bool,
            "description": "是否生成单词级时间戳",
            "constraints": "需要启用增强版语音识别"
        }
    )
    
    speech_batch_size: int = field(
        default=4,
        metadata={
            "type": int,
            "description": "语音识别批处理大小",
            "range": [1, 64],
            "constraints": "根据GPU显存调整"
        }
    )
    
    use_whisperx: bool = field(
        default=True,
        metadata={
            "type": bool,
            "description": "是否使用WhisperX替代原版Whisper",
            "constraints": "WhisperX提供更好的时间戳精度"
        }
    )
    
    whisperx_temperature: float = field(
        default=0.0,
        metadata={
            "type": float,
            "description": "WhisperX采样温度 (推荐: 0.0)",
            "range": [0.0, 1.0],
            "constraints": "0.0为确定性解码，提高准确率"
        }
    )
    
    whisperx_patience: float = field(
        default=2.0,
        metadata={
            "type": float,
            "description": "WhisperX束搜索耐心值 (推荐: 2.0)",
            "range": [1.0, 5.0],
            "constraints": "值越大搜索越充分，但速度越慢"
        }
    )
    
    whisperx_length_penalty: float = field(
        default=1.0,
        metadata={
            "type": float,
            "description": "WhisperX长度惩罚系数 (推荐: 1.0)",
            "range": [0.0, 2.0],
            "constraints": "1.0为简单长度归一化"
        }
    )
    
    whisperx_best_of: int = field(
        default=5,
        metadata={
            "type": int,
            "description": "WhisperX非零温度时的候选数 (推荐: 5)",
            "range": [1, 10],
            "constraints": "温度非零时的候选数量"
        }
    )
    
    whisperx_beam_size: int = field(
        default=5,
        metadata={
            "type": int,
            "description": "WhisperX Beam Search 的宽度（temperature 为 0 时生效）",
            "range": [1, 20],
            "constraints": "值越大识别越准确但速度越慢"
        }
    )
    
    whisperx_chunk_size: int = field(
        default=15,
        metadata={
            "type": int,
            "description": "WhisperX VAD分段大小 (推荐: 30)",
            "range": [10, 60],
            "constraints": "分段大小（秒），过大可能导致内存不足"
        }
    )
    
    whisperx_vad_onset: float = field(
        default=0.3,
        metadata={
            "type": float,
            "description": "WhisperX VAD起始阈值 (推荐: 0.3)",
            "range": [0.0, 1.0],
            "constraints": "降低此值可检测更多语音"
        }
    )
    
    whisperx_vad_offset: float = field(
        default=0.3,
        metadata={
            "type": float,
            "description": "WhisperX VAD结束阈值 (推荐: 0.3)",
            "range": [0.0, 1.0],
            "constraints": "降低此值可检测更多语音"
        }
    )
    
    whisperx_compute_type: str = field(
        default="float16",
        metadata={
            "type": str,
            "description": "WhisperX计算类型 (推荐: float16)",
            "constraints": "可选: float16, float32, int8。float16平衡速度和精度"
        }
    )
    
    whisperx_condition_on_previous_text: bool = field(
        default=False,
        metadata={
            "type": bool,
            "description": "WhisperX是否基于前文条件预测",
            "constraints": "False可减少错误累积，提高准确率"
        }
    )
    
    whisperx_suppress_numerals: bool = field(
        default=False,
        metadata={
            "type": bool,
            "description": "WhisperX是否抑制数字符号",
            "constraints": "True可提高对齐精度，但会丢失数字信息"
        }
    )
    
    whisperx_suppress_punctuation: bool = field(
        default=True,
        metadata={
            "type": bool,
            "description": "WhisperX是否抑制顿号、句号等标点符号",
            "constraints": "True可减少标点符号，使字幕更简洁"
        }
    )
    
    whisperx_suppress_tokens: str = field(
        default="-1",
        metadata={
            "type": str,
            "description": "WhisperX 抑制特定 token（如特殊字符）",
            "constraints": "-1表示不抑制任何token，多个token ID用逗号分隔，如'1231,120,234'\n" +
                          "常见特殊符号token ID：\n" +
                          "中文标点：、(1231), ，(171,120,234), 。(1543), ！(171,120,223), ？(171,120,253), ；(171,120,249), ：(171,120,248)\n" +
                          "英文标点：,(11), .(13), !(0), ?(30), ;(26), :(25), \"(1), '(6), ((7), )(8), [(58), ](60)\n" +
                          "其他符号：…(1260), —(2958), ～(171,121,252), @(31), #(2), $(3), %(4), ^(61), &(5), *(9), _(62), +(10), =(28)"
        }
    )
    
    whisperx_initial_prompt: str = field(
        default="",
        metadata={
            "type": str,
            "description": "WhisperX初始提示文本",
            "constraints": "为模型提供上下文提示，可提高特定领域识别率"
        }
    )
    
    whisperx_hotwords: str = field(
        default="",
        metadata={
            "type": str,
            "description": "WhisperX热词/提示词",
            "constraints": "专业术语，用逗号分隔，如: WhisperX, PyAnnote, GPU"
        }
    )
    
    whisperx_logprob_threshold: float = field(
        default=-1.0,
        metadata={
            "type": float,
            "description": "WhisperX对数概率阈值 (推荐: -1.0)",
            "range": [-5.0, 0.0],
            "constraints": "低于此值视为解码失败，-1.0为不限制"
        }
    )
    
    whisperx_no_speech_threshold: float = field(
        default=0.6,
        metadata={
            "type": float,
            "description": "WhisperX无语音阈值 (推荐: 0.6)",
            "range": [0.0, 1.0],
            "constraints": "高于此值且解码失败时视为静音"
        }
    )
    
    whisperx_compression_ratio_threshold: float = field(
        default=2.4,
        metadata={
            "type": float,
            "description": "WhisperX压缩比阈值 (推荐: 2.4)",
            "range": [1.0, 5.0],
            "constraints": "gzip压缩比高于此值视为解码失败"
        }
    )
    
    whisperx_temperature_increment_on_fallback: float = field(
        default=0.2,
        metadata={
            "type": float,
            "description": "WhisperX回退时温度增量 (推荐: 0.2)",
            "range": [0.0, 1.0],
            "constraints": "解码失败时增加的温度值"
        }
    )
    
    # =========================================================================
    # 3. 翻译配置 (Translation Settings)
    # =========================================================================
    
    # -------------------------------------------------------------------------
    # 翻译批处理与上下文配置 (Translation Batch Settings)
    # -------------------------------------------------------------------------
    translation_batch_size: int = field(
        default=3500,
        metadata={
            "type": int,
            "description": "翻译批处理大小（基于token数）",
            "range": [1024, 8192],
            "constraints": "每个批次的最大token数，推荐4096。根据GPU显存和llama.cpp配置调整"
        }
    )
    
    translation_context_size: int = field(
        default=8192,
        metadata={
            "type": int,
            "description": "翻译模型上下文大小",
            "range": [1024, 32768],
            "constraints": "更大的上下文可以处理更长的批量翻译，但需要更多显存"
        }
    )
    
    # -------------------------------------------------------------------------
    # 翻译采样与惩罚配置 (Translation Sampling & Penalty Settings)
    # -------------------------------------------------------------------------
    translation_temperature: float = field(
        default=0,
        metadata={
            "type": float,
            "description": "翻译模型温度参数，控制输出的随机性 (推荐字幕翻译: 0.2-0.4)",
            "range": [0.0, 2.0],
            "constraints": "字幕翻译建议0.3，平衡准确性和自然度"
        }
    )
    
    translation_top_k: int = field(
        default=20,
        metadata={
            "type": int,
            "description": "翻译模型Top-K采样参数 (推荐字幕翻译: 20-40)",
            "range": [1, 100],
            "constraints": "字幕翻译建议20，限制候选词提高准确性"
        }
    )
    
    translation_top_p: float = field(
        default=0.6,
        metadata={
            "type": float,
            "description": "翻译模型Top-P（核采样）参数 (推荐字幕翻译: 0.5-0.7)",
            "range": [0.0, 1.0],
            "constraints": "字幕翻译建议0.6，避免过于随机的输出"
        }
    )
    
    translation_min_p: float = field(
        default=0.05,
        metadata={
            "type": float,
            "description": "翻译模型Min-P采样参数 (推荐字幕翻译: 0.0-0.1)",
            "range": [0.0, 1.0],
            "constraints": "字幕翻译建议0.05，过滤低概率token提高质量"
        }
    )
    
    translation_repetition_penalty: float = field(
        default=1.05,
        metadata={
            "type": float,
            "description": "翻译模型重复惩罚参数 (推荐字幕翻译: 1.02-1.1)",
            "range": [1.0, 2.0],
            "constraints": "字幕翻译建议1.05，轻微抑制重复"
        }
    )
    
    translation_presence_penalty: float = field(
        default=0.0,
        metadata={
            "type": float,
            "description": "翻译模型存在惩罚参数 (推荐字幕翻译: 0.0-0.2)",
            "range": [0.0, 2.0],
            "constraints": "字幕翻译建议0.0-0.1，轻微惩罚重复出现的主题"
        }
    )
    
    translation_frequency_penalty: float = field(
        default=0.0,
        metadata={
            "type": float,
            "description": "翻译模型频率惩罚参数 (推荐字幕翻译: 0.0-0.2)",
            "range": [0.0, 2.0],
            "constraints": "字幕翻译建议0.0-0.1，根据出现频率惩罚重复词"
        }
    )
    
    # =========================================================================
    # 4. 增强版语音识别配置 (Enhanced Speech Recognition Settings)
    # =========================================================================
    
    # -------------------------------------------------------------------------
    # 强制对齐与段落分割配置 (Forced Alignment & Segmentation Settings)
    # -------------------------------------------------------------------------
    enable_forced_alignment: bool = field(
        default=True,
        metadata={
            "type": bool,
            "description": "是否启用强制对齐（Wav2Vec2/CTC）",
            "constraints": "可提高时间戳精度，需要下载对齐模型"
        }
    )
    
    max_segment_duration: float = field(
        default=7.0,
        metadata={
            "type": float,
            "description": "最大段落持续时间（秒）",
            "range": [1.0, 30.0],
            "constraints": "超过此值的段落会被分割"
        }
    )
    
    sentence_pause_threshold: float = field(
        default=0.5,
        metadata={
            "type": float,
            "description": "句子停顿阈值（秒）",
            "range": [0.1, 2.0],
            "constraints": "用于基于语义的断句"
        }
    )
    
    min_silence_for_split: float = field(
        default=0.3,
        metadata={
            "type": float,
            "description": "断句所需最小静音时长（秒）",
            "range": [0.1, 1.0],
            "constraints": "低于此值的静音不会触发断句"
        }
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


# =============================================================================
# 配置验证器
# =============================================================================

class ConfigValidator:
    """配置参数验证器"""
    
    @staticmethod
    def validate_type(key: str, value: Any, expected_type: type) -> Tuple[bool, str]:
        """验证参数类型"""
        if value is None:
            return True, ""
        
        if expected_type == bool:
            return (isinstance(value, bool), f"参数 {key} 类型错误: 期望 bool，实际 {type(value).__name__}")
        
        if expected_type == int:
            if isinstance(value, bool):
                return False, f"参数 {key} 不能是布尔值，应为整数"
            return (isinstance(value, int) or (isinstance(value, float) and value.is_integer()), 
                    f"参数 {key} 类型错误: 期望 int，实际 {type(value).__name__}")
        
        if expected_type == float:
            return (isinstance(value, (int, float)) and not isinstance(value, bool), 
                    f"参数 {key} 类型错误: 期望 float，实际 {type(value).__name__}")
        
        if expected_type == str:
            return (isinstance(value, str), f"参数 {key} 类型错误: 期望 str，实际 {type(value).__name__}")
        
        return True, ""
    
    @staticmethod
    def validate_range(key: str, value: Any, range_def: List[float]) -> Tuple[bool, str]:
        """验证数值范围"""
        if value is None:
            return True, ""
        
        min_val, max_val = range_def
        return (min_val <= value <= max_val, 
                f"参数 {key} 超出范围: {value} 不在 [{min_val}, {max_val}] 内")
    
    @staticmethod
    def validate_options(key: str, value: Any, options: List[Any]) -> Tuple[bool, str]:
        """验证选项值"""
        if value is None:
            return True, ""
        
        return (value in options, f"参数 {key} 值无效: {value} 不在可选值 {options} 中")
    
    @classmethod
    def validate_param(cls, key: str, value: Any, definition: Dict[str, Any]) -> Tuple[bool, str]:
        """验证单个参数"""
        # 类型验证
        if "type" in definition:
            valid, msg = cls.validate_type(key, value, definition["type"])
            if not valid:
                return False, msg
        
        # 范围验证
        if "range" in definition and value is not None:
            valid, msg = cls.validate_range(key, value, definition["range"])
            if not valid:
                return False, msg
        
        # 选项验证
        if "options" in definition and value is not None:
            valid, msg = cls.validate_options(key, value, definition["options"])
            if not valid:
                return False, msg
        
        return True, ""


# =============================================================================
# 配置管理器
# =============================================================================

class ConfigManager:
    """
    配置管理器
    提供配置的加载、保存、验证和访问功能
    """
    
    _instance = None
    _definitions = ConfigDefinitions()
    _validator = ConfigValidator()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._data = cls._definitions.to_dict()
            cls._instance._errors = []
            cls._instance._load()
        return cls._instance
    
    def _get_param_definition(self, key: str) -> Optional[Dict[str, Any]]:
        """获取参数定义"""
        if hasattr(self._definitions, key):
            field_obj = self._definitions.__dataclass_fields__.get(key)
            if field_obj:
                return field_obj.metadata
        return None
    
    def _validate_all(self) -> bool:
        """验证所有当前配置"""
        self._errors = []
        
        for key, value in self._data.items():
            definition = self._get_param_definition(key)
            if definition:
                valid, msg = self._validator.validate_param(key, value, definition)
                if not valid:
                    self._errors.append(msg)
                    # 使用默认值替换无效值
                    default_value = getattr(self._definitions, key, None)
                    if default_value is not None:
                        self._data[key] = default_value
                        print(f"[配置警告] {msg}，已重置为默认值: {default_value}")
        
        return len(self._errors) == 0
    
    def _load(self):
        """从文件加载用户配置"""
        if not os.path.exists(CONFIG_FILE):
            return
        
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                user_data = json.load(f)
            
            loaded_count = 0
            for key in self._data:
                if key in user_data:
                    definition = self._get_param_definition(key)
                    if definition:
                        valid, msg = self._validator.validate_param(key, user_data[key], definition)
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
        """
        获取配置值
        
        Args:
            key: 配置键名
            default: 默认值（如果键不存在）
            
        Returns:
            配置值
        """
        return self._data.get(key, default)
    
    def get_all(self) -> Dict[str, Any]:
        """获取完整配置副本"""
        return self._data.copy()
    
    def set(self, key: str, value: Any) -> Tuple[bool, str]:
        """
        设置单个配置值
        
        Args:
            key: 配置键名
            value: 配置值
            
        Returns:
            (是否成功, 消息)
        """
        if key not in self._data:
            return False, f"未知配置项: {key}"
        
        definition = self._get_param_definition(key)
        if definition:
            valid, msg = self._validator.validate_param(key, value, definition)
            if not valid:
                return False, msg
        
        self._data[key] = value
        return True, f"已设置 {key} = {value}"
    
    def save(self, **kwargs) -> Tuple[bool, str]:
        """保存配置到文件"""
        try:
            # 处理翻译模型映射
            if 'translator' in kwargs:
                kwargs['translator'] = TRANSLATOR_MAP.get(kwargs['translator'], kwargs['translator'])
            
            # 验证并更新配置
            for key, value in kwargs.items():
                if key in self._data:
                    valid, msg = self.set(key, value)
                    if not valid:
                        return False, msg
            
            # 保存所有参数，而不仅仅是与默认值不同的参数
            defaults = self._definitions.to_dict()
            to_save = {k: v for k, v in self._data.items() if k in defaults}
            
            # 写入文件
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(to_save, f, indent=2, ensure_ascii=False)
            
            return True, f"已保存 {len(to_save)} 个自定义参数"
            
        except Exception as e:
            return False, f"保存失败: {e}"
    
    def reset(self) -> Tuple[bool, str]:
        """重置为默认配置"""
        self._data = self._definitions.to_dict()
        self._errors = []
        
        if os.path.exists(CONFIG_FILE):
            try:
                os.remove(CONFIG_FILE)
            except Exception as e:
                return False, f"删除配置文件失败: {e}"
        
        return True, "已恢复默认配置"
    
    def build_params(self, **overrides) -> Dict[str, Any]:
        """构建处理参数"""
        params = self._data.copy()
        
        # 处理翻译模型映射
        if 'translator' in overrides:
            params['translator'] = TRANSLATOR_MAP.get(overrides['translator'], overrides['translator'])
        
        # 处理语言模式
        if 'language_mode' in overrides:
            params['language_mode'] = "auto_detect" if overrides['language_mode'] == "自动检测" else "manual"
        
        # 应用其他覆盖
        for key in self._definitions.to_dict():
            if key in overrides:
                params[key] = overrides[key]
        
        return params
    
    def get_validation_errors(self) -> List[str]:
        """获取验证错误列表"""
        return self._errors.copy()
    
    def is_valid(self) -> bool:
        """检查配置是否有效"""
        return len(self._errors) == 0
    
    def get_param_info(self, key: str) -> Optional[Dict[str, Any]]:
        """获取参数详细信息"""
        definition = self._get_param_definition(key)
        if not definition:
            return None
        
        return {
            "key": key,
            "current_value": self._data.get(key),
            "default_value": getattr(self._definitions, key),
            "type": definition.get("type").__name__,
            "description": definition.get("description"),
            "options": definition.get("options"),
            "range": definition.get("range"),
            "constraints": definition.get("constraints")
        }
    
    def list_all_params(self) -> List[Dict[str, Any]]:
        """列出所有参数信息"""
        return [self.get_param_info(key) for key in self._data.keys()]
    
    def ui_values(self) -> Tuple:
        """获取UI显示值"""
        translator = self._data['translator']
        return (
            self._data['model'],
            translator.split('/')[-1] if '/' in translator else translator,
            self._data['device'],
            self._data['source_language'],
            self._data['target_language'],
            "自动检测" if self._data['source_language'] == 'auto' else "手动选择"
        )
    
    def is_custom(self, key: str) -> bool:
        """检查参数是否被自定义"""
        defaults = self._definitions.to_dict()
        return key in defaults and self._data.get(key) != defaults[key]
    
    def get_segmentation_options(self) -> Dict[str, float]:
        """获取断句配置选项"""
        return {
            "min_silence_duration": self._data.get('min_silence_for_split', 0.3),
            "min_speech_duration": 0.25,  # 使用固定默认值
            "max_segment_duration": self._data.get('max_segment_duration', 10.0),
            "sentence_pause_threshold": self._data.get('sentence_pause_threshold', 0.5)
        }


# =============================================================================
# 常量定义
# =============================================================================

# 模型目录
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Whisper模型到目录的映射
WHISPER_MODEL_DIRS: Dict[str, str] = {
    "tiny": "tiny",
    "base": "base",
    "small": "small",
    "medium": "medium",
    "large-v2": "large-v2",
    "large-v3": "large-v3",
}

# Whisper模型目录名模式映射
WHISPER_MODEL_PATTERNS: Dict[str, List[str]] = {
    "tiny": ["tiny", "Systran--faster-whisper-tiny", "systran--faster-whisper-tiny"],
    "base": ["base", "Systran--faster-whisper-base", "systran--faster-whisper-base"],
    "small": ["small", "Systran--faster-whisper-small", "systran--faster-whisper-small"],
    "medium": ["medium", "Systran--faster-whisper-medium", "systran--faster-whisper-medium"],
    "large-v2": ["large-v2", "Systran--faster-whisper-large-v2", "systran--faster-whisper-large-v2"],
    "large-v3": ["large-v3", "Systran--faster-whisper-large-v3", "systran--faster-whisper-large-v3"],
}

def get_available_models() -> List[str]:
    """
    自动扫描models目录，返回可用的模型列表

    Returns:
        可用的模型名称列表
    """
    available_models = []

    # 扫描models目录下的所有子目录
    if os.path.exists(MODELS_DIR):
        for item in os.listdir(MODELS_DIR):
            item_path = os.path.join(MODELS_DIR, item)
            if os.path.isdir(item_path):
                item_lower = item.lower()
                # 检查是否是有效的Whisper模型目录
                for model_name, patterns in WHISPER_MODEL_PATTERNS.items():
                    if item_lower in patterns or any(p in item_lower for p in patterns):
                        if model_name not in available_models:
                            available_models.append(model_name)
                        break

    # 如果没有找到本地模型，返回默认的Whisper模型列表
    if not available_models:
        available_models = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]

    # 按推荐顺序排序
    model_priority = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]
    available_models.sort(key=lambda x: model_priority.index(x) if x in model_priority else 999)

    return available_models

# 动态获取可用的模型选项
MODEL_OPTIONS: List[str] = get_available_models()

# 翻译模型映射
TRANSLATOR_MAP: Dict[str, str] = {
    "HY-MT1.5-7B-GGUF": "tencent/HY-MT1.5-7B-GGUF"
}

# 语言选项 (代码, 显示名称)
LANGUAGE_OPTIONS: List[Tuple[str, str]] = [
    ("auto", "自动检测"),
    ("zh", "中文"),
    ("en", "英语"),
    ("ja", "日语"),
    ("ko", "韩语"),
    ("fr", "法语"),
    ("de", "德语"),
    ("es", "西班牙语"),
    ("ru", "俄语"),
    ("ar", "阿拉伯语"),
    ("hi", "印地语"),
    ("pt", "葡萄牙语"),
    ("it", "意大利语"),
    ("nl", "荷兰语"),
    ("pl", "波兰语")
]

# 对齐模型选项
ALIGNMENT_MODEL_OPTIONS: List[str] = ["wav2vec2", "ctc", "none"]

# 设备选项
DEVICE_OPTIONS: List[str] = ["auto", "cuda", "cpu"]

# 翻译模型量化选项
TRANSLATOR_QUANTIZATION_OPTIONS: List[str] = ["auto", "Q4_K_M", "Q6_K", "Q8_0"]


# =============================================================================
# 全局配置实例
# =============================================================================

# 向后兼容：保留 config 变量名
config = ConfigManager()

# 导出主要类
__all__ = [
    'ConfigManager',
    'ConfigDefinitions',
    'ConfigValidator',
    'config',
    'MODEL_OPTIONS',
    'TRANSLATOR_MAP',
    'LANGUAGE_OPTIONS',
    'ALIGNMENT_MODEL_OPTIONS',
    'DEVICE_OPTIONS',
    'TRANSLATOR_QUANTIZATION_OPTIONS',
    'PROJECT_ROOT',
    'MODEL_CACHE_DIR',
    'TEMP_DIR',
    'OUTPUT_DIR',
    'CONFIG_FILE'
]
