# -*- coding: utf-8 -*-
"""
合并后的语音识别模块
包含基础语音识别和增强版语音识别功能
"""

import os

# 强制使用 UTF-8 编码
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['LC_ALL'] = 'en_US.UTF-8'
os.environ['LANG'] = 'en_US.UTF-8'

import datetime
import gc
import subprocess
import tempfile
import wave
import contextlib
import re
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field

# 设置环境变量
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HUGGINGFACE_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TRANSFORMERS_OFFLINE"] = "1"  # 强制离线模式，禁止自动下载
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.environ["HF_HOME"] = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

import torch
import numpy as np
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import whisperx
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio")

from config import MODEL_CACHE_DIR


# 公共工具函数
def timestamp_print(message):
    """带时间戳的打印函数"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


def clear_model_cache():
    """清空模型缓存以释放内存"""
    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        timestamp_print(f"[内存管理] 清空CUDA缓存时出错: {str(e)}")
    gc.collect()
    timestamp_print("[内存管理] 已执行垃圾回收")


def check_local_model(model_name):
    """检查本地模型文件是否存在"""
    model_paths = [
        os.path.join(MODEL_CACHE_DIR, model_name),
        os.path.join(MODEL_CACHE_DIR, f"whisper-{model_name}"),
        os.path.join(MODEL_CACHE_DIR, f"openai--whisper-{model_name}"),
        os.path.join(MODEL_CACHE_DIR, f"Systran--faster-whisper-{model_name}"),
        os.path.join(MODEL_CACHE_DIR, f"models--openai--whisper-{model_name}", "snapshots"),
        os.path.join(MODEL_CACHE_DIR, f"models--Systran--faster-whisper-{model_name}", "snapshots"),
        os.path.join(MODEL_CACHE_DIR, "openai", f"whisper-{model_name}"),
        os.path.join(MODEL_CACHE_DIR, "Systran", f"faster-whisper-{model_name}")
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            if "snapshots" in path and os.path.isdir(path):
                try:
                    snapshot_dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
                    if snapshot_dirs:
                        return os.path.join(path, snapshot_dirs[0])
                except:
                    pass
            elif os.path.isdir(path):
                return path
    
    return None


def get_audio_duration(audio_path):
    """获取音频时长"""
    try:
        with contextlib.closing(wave.open(audio_path, 'r')) as f:
            return f.getnframes() / float(f.getframerate())
    except:
        return 60


def extract_middle_audio(audio_path, duration=30):
    """提取音频中间的指定时长片段"""
    audio_duration = get_audio_duration(audio_path)
    start_time = max(0, (audio_duration - duration) / 2)
    
    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    temp_file.close()
    
    try:
        subprocess.run([
            'ffmpeg', '-i', audio_path, '-ss', str(start_time), '-t', str(duration),
            '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', '-y', temp_file.name
        ], check=True, capture_output=True)
        return temp_file.name
    except Exception as e:
        timestamp_print(f"[错误信息] 提取中间音频片段时出错: {str(e)}")
        try:
            os.unlink(temp_file.name)
        except:
            pass
        return audio_path


def detect_language_from_text(text):
    """从文本内容检测语言"""
    if any('\u3040' <= char <= '\u30ff' for char in text):
        return 'ja'
    if any('\u4e00' <= char <= '\u9fff' for char in text):
        return 'zh'
    if all(ord(c) < 128 or c.isspace() for c in text):
        return 'en'
    return 'unknown'


# 增强版语音识别相关类
@dataclass
class WordTimestamp:
    """单词级时间戳数据结构"""
    word: str
    start: float
    end: float
    confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'word': self.word,
            'start': self.start,
            'end': self.end,
            'confidence': self.confidence
        }


@dataclass
class AlignedSegment:
    """对齐后的语音段数据结构"""
    id: int
    start: float
    end: float
    text: str
    words: List[WordTimestamp] = field(default_factory=list)
    speaker: Optional[str] = None
    confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'start': self.start,
            'end': self.end,
            'text': self.text,
            'words': [w.to_dict() for w in self.words],
            'speaker': self.speaker,
            'confidence': self.confidence
        }


class VADSegmenter:
    """基于 VAD 的智能断句模块"""
    
    def __init__(self, 
                 min_silence_duration: float = 0.3,
                 min_speech_duration: float = 0.25,
                 max_segment_duration: float = 10.0,
                 sentence_pause_threshold: float = 0.5):
        """
        初始化 VAD 断句器
        
        Args:
            min_silence_duration: 最小静音持续时间（秒）
            min_speech_duration: 最小语音持续时间（秒）
            max_segment_duration: 最大段落持续时间（秒）
            sentence_pause_threshold: 句子停顿阈值（秒）
        """
        self.min_silence_duration = min_silence_duration
        self.min_speech_duration = min_speech_duration
        self.max_segment_duration = max_segment_duration
        self.sentence_pause_threshold = sentence_pause_threshold
        
        # 句子结束标点符号
        self.sentence_endings = re.compile(r'[.!?。！？]+')
        
    def segment_by_vad(self, 
                       segments: List[Dict],
                       vad_segments: Optional[List[Tuple[float, float]]] = None) -> List[AlignedSegment]:
        """
        基于 VAD 的断句处理
        
        Args:
            segments: WhisperX 原始识别结果
            vad_segments: VAD 检测到的语音段 [(start, end), ...]
            
        Returns:
            断句后的对齐段落列表
        """
        aligned_segments = []
        segment_id = 0
        
        for seg in segments:
            # 检查是否需要进一步分割
            seg_duration = seg['end'] - seg['start']
            text = seg.get('text', '').strip()
            
            if seg_duration > self.max_segment_duration:
                # 长段落需要分割
                sub_segments = self._split_long_segment(seg)
                for sub_seg in sub_segments:
                    aligned_segments.append(self._create_aligned_segment(segment_id, sub_seg))
                    segment_id += 1
            else:
                # 检查是否需要基于语义断句
                if self._should_split_by_semantic(text, seg_duration):
                    sub_segments = self._split_by_semantic(seg)
                    for sub_seg in sub_segments:
                        aligned_segments.append(self._create_aligned_segment(segment_id, sub_seg))
                        segment_id += 1
                else:
                    aligned_segments.append(self._create_aligned_segment(segment_id, seg))
                    segment_id += 1
        
        return aligned_segments
    
    def _split_long_segment(self, segment: Dict) -> List[Dict]:
        """分割长语音段"""
        duration = segment['end'] - segment['start']
        text = segment.get('text', '')
        words = segment.get('words', [])
        
        if not words or len(words) < 2:
            return [segment]
        
        # 计算每个子段的目标时长
        num_subsegments = int(np.ceil(duration / self.max_segment_duration))
        target_duration = duration / num_subsegments
        
        sub_segments = []
        current_words = []
        current_start = segment['start']
        
        for word_info in words:
            word_end = word_info.get('end', word_info.get('start', 0))
            
            if word_end - current_start > target_duration and current_words:
                # 创建新子段 - 直接拼接单词，不处理空格
                text = ''.join(w['word'] for w in current_words)
                
                sub_segments.append({
                    'start': current_start,
                    'end': word_end,
                    'text': text,
                    'words': current_words.copy(),
                    'confidence': np.mean([w.get('confidence', 0) for w in current_words])
                })
                current_words = []
                current_start = word_info.get('start', word_end)
            
            current_words.append(word_info)
        
        # 添加最后一个子段 - 直接拼接单词，不处理空格
        if current_words:
            text = ''.join(w['word'] for w in current_words)
            
            sub_segments.append({
                'start': current_start,
                'end': segment['end'],
                'text': text,
                'words': current_words,
                'confidence': np.mean([w.get('confidence', 0) for w in current_words])
            })
        
        return sub_segments if sub_segments else [segment]
    
    def _should_split_by_semantic(self, text: str, duration: float) -> bool:
        """判断是否需要基于语义断句"""
        # 检查是否有明显的句子边界
        if self.sentence_endings.search(text):
            return True
        
        # 检查是否有明显的停顿标记（如逗号）
        if ',' in text or '，' in text:
            return duration > self.sentence_pause_threshold
        
        return False
    
    def _split_by_semantic(self, segment: Dict) -> List[Dict]:
        """基于语义进行断句"""
        text = segment.get('text', '')
        words = segment.get('words', [])
        
        if not words:
            return [segment]
        
        # 找到句子边界位置
        split_positions = []
        for match in self.sentence_endings.finditer(text):
            # 找到对应的单词索引
            pos = match.end()
            char_count = 0
            for i, word_info in enumerate(words):
                word = word_info.get('word', '')
                char_count += len(word) + 1  # +1 for space
                if char_count >= pos:
                    split_positions.append(i)
                    break
        
        if not split_positions:
            return [segment]
        
        # 根据边界分割 - 直接拼接单词，不处理空格
        sub_segments = []
        start_idx = 0
        
        for end_idx in split_positions:
            sub_words = words[start_idx:end_idx + 1]
            if sub_words:
                text = ''.join(w['word'] for w in sub_words)
                
                sub_segments.append({
                    'start': sub_words[0].get('start', segment['start']),
                    'end': sub_words[-1].get('end', segment['end']),
                    'text': text,
                    'words': sub_words,
                    'confidence': np.mean([w.get('confidence', 0) for w in sub_words])
                })
            start_idx = end_idx + 1
        
        # 添加剩余部分 - 直接拼接单词，不处理空格
        if start_idx < len(words):
            sub_words = words[start_idx:]
            text = ''.join(w['word'] for w in sub_words)
            
            sub_segments.append({
                'start': sub_words[0].get('start', segment['start']),
                'end': sub_words[-1].get('end', segment['end']),
                'text': text,
                'words': sub_words,
                'confidence': np.mean([w.get('confidence', 0) for w in sub_words])
            })
        
        return sub_segments if sub_segments else [segment]
    
    def _create_aligned_segment(self, segment_id: int, seg_data: Dict) -> AlignedSegment:
        """创建对齐段落对象"""
        words = []
        for w in seg_data.get('words', []):
            words.append(WordTimestamp(
                word=w.get('word', ''),
                start=w.get('start', 0.0),
                end=w.get('end', 0.0),
                confidence=w.get('confidence', 0.0)
            ))
        
        return AlignedSegment(
            id=segment_id,
            start=seg_data.get('start', 0.0),
            end=seg_data.get('end', 0.0),
            text=seg_data.get('text', ''),
            words=words,
            confidence=seg_data.get('confidence', 0.0)
        )


class ForcedAligner:
    """强制对齐模块 - 使用 Wav2Vec2/CTC 进行帧级对齐"""
    
    def __init__(self, device: str = "cuda", compute_type: str = "float16"):
        """
        初始化强制对齐器
        
        Args:
            device: 计算设备 ("cuda" 或 "cpu")
            compute_type: 计算类型 ("float16" 或 "float32")
        """
        self.device = device
        self.compute_type = compute_type
        self.align_model = None
        self.align_metadata = None
        
    def load_alignment_model(self, language_code: str) -> bool:
        """
        加载对齐模型
        
        Args:
            language_code: 语言代码 (如 "en", "zh", "ja")
            
        Returns:
            是否成功加载模型
        """
        timestamp_print(f"[强制对齐] 正在加载对齐模型 (语言: {language_code})...")
        
        # 检查本地是否存在wav2vec2模型
        wav2vec2_models = {
            "en": "jonatasgrosman--wav2vec2-large-xlsr-53-english",
            "zh": "jonatasgrosman--wav2vec2-large-xlsr-53-chinese-zh-cn",
            "ja": "jonatasgrosman--wav2vec2-large-xlsr-53-japanese",
            "fr": "jonatasgrosman--wav2vec2-large-xlsr-53-french",
            "de": "jonatasgrosman--wav2vec2-large-xlsr-53-german",
            "es": "jonatasgrosman--wav2vec2-large-xlsr-53-spanish",
            "ru": "jonatasgrosman--wav2vec2-large-xlsr-53-russian",
            "ar": "jonatasgrosman--wav2vec2-large-xlsr-53-arabic",
            "pt": "jonatasgrosman--wav2vec2-large-xlsr-53-portuguese",
            "it": "jonatasgrosman--wav2vec2-large-xlsr-53-italian",
            "nl": "jonatasgrosman--wav2vec2-large-xlsr-53-dutch",
            "pl": "jonatasgrosman--wav2vec2-large-xlsr-53-polish",
            "fi": "jonatasgrosman--wav2vec2-large-xlsr-53-finnish",
            "fa": "jonatasgrosman--wav2vec2-large-xlsr-53-persian"
        }
        
        model_exists = False
        if language_code in wav2vec2_models:
            model_name = wav2vec2_models[language_code]
            model_dir = os.path.join(MODEL_CACHE_DIR, model_name)
            if os.path.exists(model_dir):
                timestamp_print(f"[强制对齐] 找到本地wav2vec2模型: {model_dir}")
                model_exists = True
            else:
                timestamp_print(f"[强制对齐警告] 本地wav2vec2模型不存在: {model_dir}")
        else:
            timestamp_print(f"[强制对齐警告] 语言 {language_code} 没有对应的wav2vec2模型")
        
        try:
            if model_exists:
                # 检查本地模型目录结构
                model_name = wav2vec2_models[language_code]
                model_path = os.path.join(MODEL_CACHE_DIR, model_name)
                timestamp_print(f"[强制对齐] 尝试加载本地模型: {model_path}")
                
                # 检查模型文件是否存在
                required_files = ['config.json', 'preprocessor_config.json', 'pytorch_model.bin', 'vocab.json']
                missing_files = [f for f in required_files if not os.path.exists(os.path.join(model_path, f))]
                if missing_files:
                    timestamp_print(f"[强制对齐警告] 缺少必要的模型文件: {missing_files}")
                    return False
                
                # 直接使用模型路径，绕过whisperx的模型下载逻辑
                import whisperx
                
                timestamp_print(f"[强制对齐] 尝试使用whisperx加载本地模型: {model_path}")
                
                # 使用whisperx的load_align_model函数，同时指定language_code和model_name参数
                self.align_model, self.align_metadata = whisperx.load_align_model(
                    language_code=language_code,
                    model_name=model_path,
                    device=self.device,
                    model_dir=MODEL_CACHE_DIR
                )
                
                timestamp_print(f"[强制对齐] 对齐模型加载完成")
                return True
            else:
                timestamp_print(f"[强制对齐警告] 本地模型不存在，跳过强制对齐")
                self.align_model = None
                self.align_metadata = None
                return False
            
        except Exception as e:
            timestamp_print(f"[强制对齐警告] 无法加载对齐模型: {str(e)}")
            timestamp_print(f"[强制对齐警告] 将使用基础语音识别结果（时间戳精度可能降低）")
            self.align_model = None
            self.align_metadata = None
            return False
        
    def align(self, 
              transcript_segments: List[Dict],
              audio_path: str,
              return_char_alignments: bool = False) -> List[Dict]:
        """
        执行强制对齐
        
        Args:
            transcript_segments: 转录文本段列表
            audio_path: 音频文件路径
            return_char_alignments: 是否返回字符级对齐
            
        Returns:
            对齐后的段落列表，包含精确的时间戳
        """
        if self.align_model is None:
            raise RuntimeError("对齐模型未加载，请先调用 load_alignment_model()")
        
        timestamp_print(f"[强制对齐] 开始帧级对齐处理...")
        
        try:
            aligned_result = whisperx.align(
                transcript_segments,
                self.align_model,
                self.align_metadata,
                audio_path,
                self.device,
                return_char_alignments=return_char_alignments
            )
            
            timestamp_print(f"[强制对齐] 对齐完成，共 {len(aligned_result.get('segments', []))} 个段落")
            
            return aligned_result.get('segments', [])
            
        except Exception as e:
            timestamp_print(f"[强制对齐错误] {str(e)}")
            # 对齐失败时返回原始段
            return transcript_segments
    
    def cleanup(self):
        """清理对齐模型资源"""
        if self.align_model is not None:
            del self.align_model
            self.align_model = None
            self.align_metadata = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class EnhancedSpeechRecognizer:
    """增强版语音识别器 - 集成对齐、断句和单词级时间戳"""
    
    def __init__(self,
                 model_path: str,
                 device: str = "cuda",
                 compute_type: str = "float16",
                 vad_options: Optional[Dict] = None,
                 enable_alignment: bool = True,
                 enable_segmentation: bool = True,
                 segmentation_options: Optional[Dict] = None):
        """
        初始化增强版语音识别器
        
        Args:
            model_path: Whisper 模型路径
            device: 计算设备
            compute_type: 计算类型
            vad_options: VAD 配置选项
            enable_alignment: 是否启用强制对齐
            enable_segmentation: 是否启用智能断句
            segmentation_options: 断句配置选项
        """
        self.model_path = model_path
        self.device = device
        self.compute_type = compute_type
        self.enable_alignment = enable_alignment
        self.enable_segmentation = enable_segmentation
        
        # VAD 配置
        default_vad_options = {
            "vad_onset": 0.3,
            "vad_offset": 0.3,
            "chunk_size": 30
        }
        self.vad_options = {**default_vad_options, **(vad_options or {})}
        
        # 断句器配置
        default_seg_options = {
            "min_silence_duration": 0.3,
            "min_speech_duration": 0.25,
            "max_segment_duration": 10.0,
            "sentence_pause_threshold": 0.5
        }
        self.segmentation_options = {**default_seg_options, **(segmentation_options or {})}
        
        # 初始化组件
        self.whisper_model = None
        self.aligner = None
        
        if self.enable_alignment:
            self.aligner = ForcedAligner(device, compute_type)
    
    def _parse_suppress_tokens(self, suppress_tokens_str: str) -> list:
        """
        解析 suppress_tokens 字符串，支持多个 token ID
        
        Args:
            suppress_tokens_str: 逗号分隔的 token ID 字符串，如 "1231,1232" 或 "-1"
            
        Returns:
            解析后的 token ID 列表
        """
        if suppress_tokens_str == "-1":
            return [-1]
        
        try:
            tokens = [int(token.strip()) for token in suppress_tokens_str.split(",")]
            return tokens
        except ValueError:
            # 如果解析失败，返回默认值
            return [-1]
    
    def load_model(self):
        """加载 WhisperX 模型"""
        timestamp_print(f"[模型加载] 正在加载 WhisperX 模型...")
        
        # 从配置获取WhisperX参数
        from config import config
        
        # ASR选项：使用配置文件中的参数
        asr_options = {
            "beam_size": config.get('whisperx_beam_size', 5),
            "patience": config.get('whisperx_patience', 2.0),
            "length_penalty": config.get('whisperx_length_penalty', 1.0),
            "best_of": config.get('whisperx_best_of', 5),
            "compression_ratio_threshold": config.get('whisperx_compression_ratio_threshold', 2.4),
            "log_prob_threshold": config.get('whisperx_logprob_threshold', -1.0),
            "no_speech_threshold": config.get('whisperx_no_speech_threshold', 0.6),
            "condition_on_previous_text": config.get('whisperx_condition_on_previous_text', False),
            "suppress_numerals": config.get('whisperx_suppress_numerals', False),
            "suppress_tokens": self._parse_suppress_tokens(config.get('whisperx_suppress_tokens', "-1")),

            "initial_prompt": config.get('whisperx_initial_prompt', None) or None,
            "hotwords": config.get('whisperx_hotwords', None) or None,
        }
        
        # 过滤掉None值的选项
        asr_options = {k: v for k, v in asr_options.items() if v is not None}
        
        # 更新VAD选项
        vad_options = {
            "vad_onset": config.get('whisperx_vad_onset', 0.3),
            "vad_offset": config.get('whisperx_vad_offset', 0.3),
            "chunk_size": config.get('whisperx_chunk_size', 30),
        }
        self.vad_options = {**self.vad_options, **vad_options}
        
        # 强制使用float32计算类型以避免设备兼容性问题
        compute_type = "float32"
        timestamp_print(f"[模型加载] 强制使用计算类型: {compute_type}")
        
        self.whisper_model = whisperx.load_model(
            self.model_path,
            device=self.device,
            compute_type=compute_type,
            vad_options=self.vad_options,
            asr_options=asr_options
        )
        
        timestamp_print(f"[模型加载] WhisperX 模型加载完成")
    
    def transcribe(self,
                   audio_path: str,
                   language: Optional[str] = None,
                   batch_size: int = 8,
                   word_timestamps: bool = True,
                   progress_callback: Optional[Callable[[int], None]] = None) -> Dict[str, Any]:
        """
        执行增强版语音识别
        
        Args:
            audio_path: 音频文件路径
            language: 语言代码（None 表示自动检测）
            batch_size: 批处理大小
            word_timestamps: 是否生成单词级时间戳
            progress_callback: 进度回调函数
            
        Returns:
            包含对齐、断句和单词级时间戳的完整结果
        """
        if self.whisper_model is None:
            self.load_model()
        
        if progress_callback:
            progress_callback(10)
        
        # 步骤 1: 基础语音识别
        timestamp_print(f"[语音识别] 执行基础识别...")
        result = self.whisper_model.transcribe(
            audio_path,
            language=language,
            batch_size=batch_size
        )
        
        if progress_callback:
            progress_callback(40)
        
        detected_language = result.get('language', 'en')
        raw_segments = result.get('segments', [])
        
        timestamp_print(f"[语音识别] 基础识别完成，检测到语言: {detected_language}")
        
        # 步骤 2: 强制对齐（如果启用）
        if self.enable_alignment and raw_segments and self.aligner:
            alignment_loaded = False
            if self.aligner.align_model is None:
                alignment_loaded = self.aligner.load_alignment_model(detected_language)
            else:
                alignment_loaded = True
            
            # 只有成功加载对齐模型才进行对齐
            if alignment_loaded:
                aligned_segments = self.aligner.align(
                    raw_segments,
                    audio_path,
                    return_char_alignments=False
                )
            else:
                # 对齐模型加载失败，使用原始段
                aligned_segments = raw_segments
                self.enable_alignment = False  # 标记对齐未成功
            
            if progress_callback:
                progress_callback(70)
        else:
            aligned_segments = raw_segments
        
        # 步骤 3: 智能断句（如果启用）
        if self.enable_segmentation:
            # 使用WhisperX的内置断句功能
            timestamp_print("[语音识别] 使用WhisperX内置断句功能...")
            final_segments = aligned_segments if aligned_segments else []
            
            if progress_callback:
                progress_callback(90)
        else:
            # 直接使用原始片段
            final_segments = aligned_segments if aligned_segments else []
        
        # 构建最终结果 - 直接拼接文本，不处理空格
        full_text = ''.join(seg['text'] if isinstance(seg, dict) else seg.text for seg in final_segments)
        
        # 构建segments列表
        segments_list = []
        for seg in final_segments:
            if isinstance(seg, dict):
                segments_list.append(seg)
            else:
                segments_list.append(seg.to_dict())
        
        result_data = {
            'text': full_text,
            'language': detected_language,
            'segments': segments_list,
            'word_timestamps_enabled': word_timestamps,
            'alignment_enabled': self.enable_alignment,
            'segmentation_enabled': self.enable_segmentation
        }
        
        if progress_callback:
            progress_callback(100)
        
        timestamp_print(f"[语音识别] 处理完成，共 {len(final_segments)} 个段落")
        
        return result_data
    
    def cleanup(self):
        """清理所有资源"""
        if self.whisper_model is not None:
            del self.whisper_model
            self.whisper_model = None
        
        if self.aligner:
            self.aligner.cleanup()
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# 主要函数
def recognize_speech(audio_path, model_path, detected_language=None, device_choice="auto",
                    progress_callback=None, vad_filter=True, word_timestamps=False,
                    condition_on_previous_text=False, use_whisperx=True, whisperx_batch_size=8,
                    vad_threshold=0.5, vad_min_speech_duration_ms=250, vad_max_speech_duration_s=30,
                    vad_min_silence_duration_ms=100):
    """使用WhisperX模型进行语音识别"""
    timestamp_print("[语音识别] 使用 WhisperX 批处理功能进行语音识别")
    timestamp_print(f"[VAD参数] VAD阈值: {vad_threshold}, 最小语音持续时间: {vad_min_speech_duration_ms}ms")
    
    device = "cuda" if device_choice != "cpu" else "cpu"
    
    # 检查本地模型
    local_model_path = check_local_model(model_path)
    if not local_model_path:
        error_msg = f"本地模型不存在: {model_path}"
        timestamp_print(f"[错误信息] {error_msg}")
        raise FileNotFoundError(error_msg)
    
    # 加载模型
    timestamp_print(f"[模型加载] 正在加载 WhisperX 模型 {model_path}...")
    vad_options = {
        "vad_onset": vad_threshold,
        "vad_offset": vad_threshold - 0.137,
        "chunk_size": 30
    }
    # ASR选项：降低压缩比阈值以更早检测重复/垃圾输出（日语等CJK语言需要更严格）
    asr_options = {
        "compression_ratio_threshold": 2.1,  # 默认2.4，降低以更早检测重复
        "log_prob_threshold": -1.0,
        "no_speech_threshold": 0.6,
    }
    # 强制使用float32计算类型以避免设备兼容性问题
    compute_type = "float32"
    timestamp_print(f"[设备配置] 使用计算类型: {compute_type}")
    
    model = whisperx.load_model(
        local_model_path,
        device=device,
        compute_type=compute_type,
        vad_options=vad_options,
        asr_options=asr_options
    )
    timestamp_print(f"[模型加载] WhisperX 模型已成功加载")
    
    if progress_callback:
        progress_callback(10)
    
    # 获取音频时长
    audio_duration = get_audio_duration(audio_path)
    timestamp_print(f"[语音识别] 音频时长: {audio_duration:.2f} 秒")
    
    # 语言检测
    if not detected_language:
        timestamp_print("[语音识别] 自动检测语言，使用音频中间30秒进行检测...")
        middle_audio = extract_middle_audio(audio_path, 30)
        lang_result = model.transcribe(middle_audio, language=None, batch_size=whisperx_batch_size)
        detected_language = lang_result.get('language', 'unknown')
        timestamp_print(f"[语音识别] 检测到的语言: {detected_language}")
        
        if middle_audio != audio_path:
            try:
                os.unlink(middle_audio)
            except:
                pass
    
    # 语音识别
    timestamp_print("[语音识别] 执行语音识别...")
    result = model.transcribe(
        audio_path,
        language=detected_language if detected_language else None,
        batch_size=whisperx_batch_size
    )
    
    if progress_callback:
        progress_callback(80)
    
    # 获取语言信息
    language = result.get('language', 'unknown')
    if language == 'unknown' and result.get('segments'):
        text = ''.join(seg.get('text', '') for seg in result['segments'])
        language = detect_language_from_text(text)
    
    timestamp_print(f"[语音识别] 检测到的语言: {language}")
    
    # 转换结果格式
    final_result = {
        'text': ''.join(seg.get('text', '') for seg in result.get('segments', [])),
        'segments': [],
        'language': language
    }
    
    for i, segment in enumerate(result.get('segments', [])):
        final_result['segments'].append({
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
        })
    
    if progress_callback:
        progress_callback(100)
    
    # 清理内存
    del model
    gc.collect()
    
    return final_result


def recognize_speech_enhanced(audio_path: str,
                               model_path: str,
                               detected_language: Optional[str] = None,
                               device_choice: str = "auto",
                               progress_callback: Optional[Callable[[int], None]] = None,
                               word_timestamps: bool = True,
                               whisperx_batch_size: int = 8,
                               vad_threshold: float = 0.5,
                               enable_alignment: bool = True,
                               enable_segmentation: bool = True,
                               segmentation_options: Optional[Dict] = None) -> Dict[str, Any]:
    """
    增强版语音识别函数（向后兼容的接口）
    
    Args:
        audio_path: 音频文件路径
        model_path: 模型路径或名称
        detected_language: 检测到的语言（None 表示自动检测）
        device_choice: 设备选择 ("auto", "cuda", "cpu")
        progress_callback: 进度回调函数
        word_timestamps: 是否启用单词级时间戳
        whisperx_batch_size: WhisperX 批处理大小
        vad_threshold: VAD 阈值
        enable_alignment: 是否启用强制对齐
        enable_segmentation: 是否启用智能断句
        segmentation_options: 断句配置选项
        
    Returns:
        包含对齐、断句和单词级时间戳的识别结果
    """
    device = "cuda" if device_choice != "cpu" and torch.cuda.is_available() else "cpu"
    # 强制使用float32计算类型以避免设备兼容性问题
    compute_type = "float32"
    timestamp_print(f"[设备配置] 使用设备: {device}, 计算类型: {compute_type}")
    
    # 检查本地模型
    local_model_path = check_local_model(model_path)
    if not local_model_path:
        error_msg = f"本地模型不存在: {model_path}"
        timestamp_print(f"[错误信息] {error_msg}")
        raise FileNotFoundError(error_msg)
    
    # VAD 配置
    vad_options = {
        "vad_onset": vad_threshold,
        "vad_offset": vad_threshold - 0.137,
        "chunk_size": 30
    }
    
    # 创建增强版识别器
    recognizer = EnhancedSpeechRecognizer(
        model_path=local_model_path,
        device=device,
        compute_type=compute_type,
        vad_options=vad_options,
        enable_alignment=enable_alignment,
        enable_segmentation=enable_segmentation,
        segmentation_options=segmentation_options
    )
    
    try:
        # 执行识别
        result = recognizer.transcribe(
            audio_path=audio_path,
            language=detected_language,
            batch_size=whisperx_batch_size,
            word_timestamps=word_timestamps,
            progress_callback=progress_callback
        )
        
        return result
        
    finally:
        # 确保资源被清理
        recognizer.cleanup()
