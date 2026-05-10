# -*- coding: utf-8 -*-
"""
语音识别模块
包含基础语音识别和强制对齐功能

"""

import os
import gc
import subprocess
import tempfile
import re
import torch
import numpy as np
import torchaudio
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from config import MODEL_CACHE_DIR, VadParams, CdParams
from utils.logger import timestamp_print

try:
    from silero_vad import get_speech_timestamps, read_audio
    SILERO_VAD_AVAILABLE = True
except ImportError:
    SILERO_VAD_AVAILABLE = False


# 公共工具函数


def check_local_model(model_name):
    """检查本地模型文件是否存在"""
    model_paths = [
        os.path.join(MODEL_CACHE_DIR, model_name),
        os.path.join(MODEL_CACHE_DIR, f"openai--whisper-{model_name}"),
        os.path.join(MODEL_CACHE_DIR, f"models--openai--whisper-{model_name}", "snapshots"),
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


class ForcedAligner:
    """强制对齐模块 - 使用 torchaudio 和 Wav2Vec2 进行帧级对齐"""
    
    def __init__(self, device: str = "auto", compute_type: str = "float16"):
        """
        初始化强制对齐器
        
        Args:
            device: 计算设备 ("auto", "cuda" 或 "cpu")
            compute_type: 计算类型 ("float16" 或 "float32")
        """
        # 自动检测设备
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.compute_type = compute_type
        self.align_model = None
        self.align_processor = None
        
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
            "ja": "jonatasgrosman--wav2vec2-large-xlsr-53-japanese",
            "zh": "jonatasgrosman--wav2vec2-large-xlsr-53-chinese-zh-cn",
        }
        
        model_exists = False
        model_path = None
        if language_code in wav2vec2_models:
            model_name = wav2vec2_models[language_code]
            model_path = os.path.join(MODEL_CACHE_DIR, model_name)
            if os.path.exists(model_path):
                timestamp_print(f"[强制对齐] 找到本地wav2vec2模型: {model_path}")
                model_exists = True
            else:
                timestamp_print(f"[强制对齐警告] 本地wav2vec2模型不存在: {model_path}")
        else:
            timestamp_print(f"[强制对齐警告] 语言 {language_code} 没有对应的wav2vec2模型")
        
        try:
            if model_exists:
                # 检查模型文件是否存在
                required_files = ['config.json', 'preprocessor_config.json', 'pytorch_model.bin', 'vocab.json']
                missing_files = [f for f in required_files if not os.path.exists(os.path.join(model_path, f))]
                if missing_files:
                    timestamp_print(f"[强制对齐警告] 缺少必要的模型文件: {missing_files}")
                    return False
                
                # 使用 torchaudio 和 transformers 加载模型
                from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
                
                # 加载处理器和模型
                self.align_processor = Wav2Vec2Processor.from_pretrained(model_path)
                self.align_model = Wav2Vec2ForCTC.from_pretrained(model_path)
                self.align_model.to(self.device)
                self.align_model.eval()
                
                timestamp_print(f"[强制对齐] 对齐模型加载完成")
                return True
            else:
                timestamp_print(f"[强制对齐警告] 本地模型不存在，跳过强制对齐")
                self.align_model = None
                self.align_processor = None
                return False

        except Exception as e:
            timestamp_print(f"[强制对齐警告] 无法加载对齐模型: {str(e)}")
            self.align_model = None
            self.align_processor = None
            return False

    def _number_to_words(self, number: int, language: str) -> str:
        """
        将数字转换为指定语言的文字

        Args:
            number: 要转换的数字
            language: 语言代码

        Returns:
            数字对应的文字表示
        """
        if language == 'en':
            ones = ['', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
                   'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen',
                   'seventeen', 'eighteen', 'nineteen']
            tens = ['', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety']

            if number < 20:
                return ones[number]
            elif number < 100:
                return tens[number // 10] + ('' if number % 10 == 0 else ' ' + ones[number % 10])
            elif number < 1000:
                return ones[number // 100] + ' hundred' + ('' if number % 100 == 0 else ' ' + self._number_to_words(number % 100, language))
            else:
                return str(number)

        elif language == 'ja':
            digits = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九']
            if number < 10:
                return digits[number]
            elif number < 20:
                return '十' + ('' if number == 10 else digits[number - 10])
            elif number < 100:
                ten = number // 10
                one = number % 10
                return digits[ten] + '十' + ('' if one == 0 else digits[one])
            elif number < 1000:
                hundred = number // 100
                rest = number % 100
                return digits[hundred] + '百' + ('' if rest == 0 else self._number_to_words(rest, language))
            else:
                return str(number)

        else:
            return str(number)

    def _preprocess_text_for_alignment(self, text: str, language: str = 'en') -> str:
        import re

        if not text:
            return ""

        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'http[s]?://\S+', '', text)

        if language in ['ja', 'zh', 'ko']:
            text = re.sub(r'[^\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff\uac00-\ud7af\u0041-\u005a\u0061-\u007a\u0030-\u0039\s]', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
        else:
            def replace_number(match):
                number_str = match.group(0)
                try:
                    number = int(number_str)
                    return self._number_to_words(number, language)
                except:
                    return number_str

            text = re.sub(r'\b\d+\b', replace_number, text)
            text = re.sub(r'[^\w\s]', '', text)

        return text

    def align(
              self,
              transcript_segments: list,
              audio_path: str,
              return_char_alignments: bool = False) -> list:
        """
        执行强制对齐

        Args:
            transcript_segments: 转录文本段列表
            audio_path: 音频文件路径
            return_char_alignments: 是否返回字符级对齐

        Returns:
            对齐后的段落列表，包含精确的时间戳
        """
        if self.align_model is None or self.align_processor is None:
            raise RuntimeError("对齐模型未加载，请先调用 load_alignment_model()")

        timestamp_print(f"[强制对齐] 开始帧级对齐处理...")

        try:
            aligned_segments = []
            sr = 16000

            timestamp_print(f"[强制对齐] 开始处理 {len(transcript_segments)} 个段落")

            for i, segment in enumerate(transcript_segments):
                text = segment.get('text', '')
                start_time = segment.get('start', 0.0)
                end_time = segment.get('end', 0.0)

                language = segment.get('language', 'en')

                timestamp_print(f"[强制对齐] 处理段落 {i+1}/{len(transcript_segments)}")

                original_text = text
                text = self._preprocess_text_for_alignment(text, language)

                if not text:
                    timestamp_print(f"[强制对齐] 跳过空文本段落")
                    aligned_segments.append(segment)
                    continue

                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    temp_audio_path = temp_file.name

                try:
                    duration = end_time - start_time
                    subprocess.run([
                        'ffmpeg', '-y', '-i', audio_path,
                        '-ss', str(start_time), '-t', str(duration),
                        '-ar', str(sr), '-ac', '1', '-acodec', 'pcm_s16le',
                        temp_audio_path
                    ], check=True, capture_output=True)

                    waveform, sample_rate = torchaudio.load(temp_audio_path)

                    if waveform.shape[0] > 1:
                        waveform = torch.mean(waveform, dim=0, keepdim=True)

                    segment_audio = waveform.squeeze().numpy()

                except Exception as e:
                    timestamp_print(f"[强制对齐警告] 提取音频片段失败: {str(e)}")
                    aligned_segments.append(segment)
                    continue
                finally:
                    try:
                        os.unlink(temp_audio_path)
                    except:
                        pass

                if len(segment_audio) == 0:
                    timestamp_print(f"[强制对齐] 音频片段为空，跳过")
                    aligned_segments.append(segment)
                    continue

                inputs = self.align_processor(segment_audio, sampling_rate=sr, return_tensors="pt")
                input_values = inputs.input_values.to(self.device)

                max_length = 16000 * 30
                if input_values.shape[1] > max_length:
                    num_chunks = (input_values.shape[1] + max_length - 1) // max_length
                    timestamp_print(f"[强制对齐] 音频过长，分 {num_chunks} 块处理")
                    logits = []
                    for chunk_idx in range(num_chunks):
                        start_idx = chunk_idx * max_length
                        end_idx = min((chunk_idx + 1) * max_length, input_values.shape[1])
                        chunk_input = input_values[:, start_idx:end_idx]

                        with torch.no_grad():
                            chunk_logits = self.align_model(chunk_input).logits
                            logits.append(chunk_logits)

                        torch.cuda.empty_cache()

                    logits = torch.cat(logits, dim=1)
                else:
                    if input_values.shape[1] < 100:
                        logits = torch.zeros((1, 1, self.align_model.config.vocab_size), device=self.device)
                        timestamp_print(f"[强制对齐] 输入太短，使用空 logits")
                    else:
                        with torch.no_grad():
                            logits = self.align_model(input_values).logits

                torch.cuda.empty_cache()

                tokens = self.align_processor.tokenizer.tokenize(text)
                if not tokens:
                    aligned_segments.append(segment)
                    continue
                labels = [self.align_processor.tokenizer.convert_tokens_to_ids(token) for token in tokens]
                labels = torch.tensor([labels], device=self.device)

                if labels.shape[1] == 0:
                    aligned_segments.append(segment)
                    continue

                input_length = logits.shape[1]
                target_length = labels.shape[1]
                max_target_length = input_length // 2

                if target_length > max_target_length:
                    split_size = max_target_length - 10
                    all_token_alignments = []
                    timestamp_print(f"[强制对齐] 段落过长，分 {(target_length + split_size - 1) // split_size} 段处理")
                    
                    for split_start in range(0, target_length, split_size):
                        split_end = min(split_start + split_size, target_length)
                        split_labels = labels[:, split_start:split_end]
                        split_target_length = split_end - split_start
                        
                        ratio_start = split_start / target_length
                        ratio_end = split_end / target_length
                        logits_start = int(input_length * ratio_start)
                        logits_end = int(input_length * ratio_end)
                        margin = min(100, (logits_end - logits_start) // 4)
                        logits_start = max(0, logits_start - margin)
                        logits_end = min(input_length, logits_end + margin)
                        
                        split_logits = logits[:, logits_start:logits_end]
                        split_input_length = split_logits.shape[1]
                        
                        try:
                            from torchaudio.functional import forced_align
                            split_paths, _ = forced_align(
                                split_logits,
                                split_labels,
                                input_lengths=torch.tensor([split_input_length], device=self.device),
                                target_lengths=torch.tensor([split_target_length], device=self.device),
                                blank=self.align_processor.tokenizer.pad_token_id
                            )
                            
                            blank_id = self.align_processor.tokenizer.pad_token_id
                            for t in range(split_paths.shape[1]):
                                token_id = split_paths[0, t].item()
                                if token_id != blank_id:
                                    token = self.align_processor.tokenizer.convert_ids_to_tokens([token_id])[0]
                                    if token not in ['[PAD]', '[UNK]', '<pad>', '<unk>']:
                                        all_token_alignments.append((t + logits_start, token))
                        except Exception as e:
                            timestamp_print(f"[强制对齐] 子段对齐失败: {str(e)}")
                    
                    token_alignments = all_token_alignments
                else:
                    input_lengths = torch.tensor([input_length], device=self.device)
                    target_lengths = torch.tensor([target_length], device=self.device)

                    from torchaudio.functional import forced_align
                    paths, scores = forced_align(
                        logits,
                        labels,
                        input_lengths=input_lengths,
                        target_lengths=target_lengths,
                        blank=self.align_processor.tokenizer.pad_token_id
                    )

                    blank_id = self.align_processor.tokenizer.pad_token_id
                    token_alignments = []
                    for t in range(paths.shape[1]):
                        token_id = paths[0, t].item()
                        if token_id != blank_id:
                            token = self.align_processor.tokenizer.convert_ids_to_tokens([token_id])[0]
                            if token not in ['[PAD]', '[UNK]', '<pad>', '<unk>']:
                                token_alignments.append((t, token))

                frame_duration = (end_time - start_time) / logits.shape[1]

                if return_char_alignments:
                    chars = []
                    for t, token in token_alignments:
                        token = token.replace('▁', ' ').strip()
                        if token:
                            char_start = start_time + t * frame_duration
                            char_end = start_time + (t + 1) * frame_duration
                            for char in token:
                                if char not in ['[PAD]', '[UNK]', '|', '<pad>', '<unk>']:
                                    chars.append({
                                        'char': char,
                                        'start': char_start,
                                        'end': char_end
                                    })
                    segment['chars'] = chars
                    timestamp_print(f"[强制对齐] 字符级对齐完成，字符数: {len(chars)}")

                if 'chars' in segment and segment['chars']:
                    words = []
                    current_word = None
                    current_word_start = None
                    current_word_end = None

                    for char_info in segment['chars']:
                        char = char_info.get('char', '')
                        char_start = char_info.get('start', 0)
                        char_end = char_info.get('end', 0)

                        if char and char != ' ':
                            if current_word is None:
                                current_word = char
                                current_word_start = char_start
                                current_word_end = char_end
                            else:
                                current_word += char
                                current_word_end = char_end
                        else:
                            if current_word is not None:
                                words.append({
                                    'word': current_word,
                                    'start': current_word_start,
                                    'end': current_word_end
                                })
                                current_word = None
                                current_word_start = None
                                current_word_end = None

                    if current_word is not None:
                        words.append({
                            'word': current_word,
                            'start': current_word_start,
                            'end': current_word_end
                        })

                    segment['words'] = words
                    timestamp_print(f"[强制对齐] 单词级对齐完成，单词数: {len(words)}")

                audio_duration = end_time - start_time
                if 'chars' in segment:
                    segment['chars'] = [c for c in segment['chars'] 
                                        if c.get('start', 0) >= 0 
                                        and c.get('end', 0) > c.get('start', 0)
                                        and c.get('end', 0) <= audio_duration + start_time + 0.1]
                if 'words' in segment:
                    segment['words'] = [w for w in segment['words'] 
                                        if w.get('start', 0) >= 0 
                                        and w.get('end', 0) > w.get('start', 0)
                                        and w.get('end', 0) <= audio_duration + start_time + 0.1]

                aligned_segments.append(segment)
                timestamp_print(f"[强制对齐] 段落 {i+1} 处理完成")

            timestamp_print(f"[强制对齐] 所有段落处理完成，共 {len(aligned_segments)} 个段落")

            return aligned_segments

        except Exception as e:
            timestamp_print(f"[强制对齐错误] {str(e)}")
            import traceback
            timestamp_print(f"[强制对齐错误详情] {traceback.format_exc()}")
            # 对齐失败时返回原始段
            return transcript_segments
    
    def cleanup(self):
        """清理对齐模型资源"""
        if self.align_model is not None:
            del self.align_model
            self.align_model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()




def _process_cd_segments(cd_result, speech_segments=None):
    """处理 Whisper-CD 结果的共享函数"""
    result = {
        'segments': [],
        'text': '',
        'language': cd_result['language']
    }
    if speech_segments is not None:
        result['vad_segments'] = speech_segments

    for segment in cd_result['segments']:
        if isinstance(segment, dict):
            start = segment.get('start', 0)
            end = segment.get('end', 0)
            text = segment.get('text', '')
            words = segment.get('words', [])
            chars = segment.get('chars', [])
        else:
            start = getattr(segment, 'start', 0)
            end = getattr(segment, 'end', 0)
            text = getattr(segment, 'text', '')
            words = getattr(segment, 'words', [])
            chars = getattr(segment, 'chars', [])
        
        processed_words = []
        for word in words:
            if isinstance(word, dict):
                processed_words.append({
                    'word': word.get('word', ''),
                    'start': word.get('start', 0),
                    'end': word.get('end', 0)
                })
            else:
                processed_words.append({
                    'word': getattr(word, 'word', ''),
                    'start': getattr(word, 'start', 0),
                    'end': getattr(word, 'end', 0)
                })

        processed_chars = []
        for char in chars:
            if isinstance(char, dict):
                char_data = {
                    'char': char.get('char', ''),
                    'start': char.get('start', 0),
                    'end': char.get('end', 0)
                }
                if 'original_logits' in char:
                    logits_val = char.get('original_logits', 0)
                    if hasattr(logits_val, 'flatten'):
                        logits_val = float(logits_val.flatten()[0]) if logits_val.size > 0 else 0.0
                    elif hasattr(logits_val, 'item'):
                        logits_val = logits_val.item()
                    elif hasattr(logits_val, '__float__'):
                        logits_val = float(logits_val)
                    char_data['original_logits'] = logits_val
                processed_chars.append(char_data)
            else:
                char_data = {
                    'char': getattr(char, 'char', ''),
                    'start': getattr(char, 'start', 0),
                    'end': getattr(char, 'end', 0)
                }
                if hasattr(char, 'original_logits'):
                    logits_val = getattr(char, 'original_logits', 0)
                    if hasattr(logits_val, 'flatten'):
                        logits_val = float(logits_val.flatten()[0]) if logits_val.size > 0 else 0.0
                    elif hasattr(logits_val, 'item'):
                        logits_val = logits_val.item()
                    elif hasattr(logits_val, '__float__'):
                        logits_val = float(logits_val)
                    char_data['original_logits'] = logits_val
                processed_chars.append(char_data)

        result['segments'].append({
            'start': start,
            'end': end,
            'text': text,
            'words': processed_words,
            'chars': processed_chars,
            'language': cd_result.get('language', 'en'),
            'temperature': 0.0,
            'avg_logprob': 0.0,
            'compression_ratio': 1.0,
            'no_speech_prob': 0.0
        })
        result['text'] += text

    return result


def _cleanup_memory():
    """清理内存和显存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def apply_silero_vad(audio_path, vad_params: VadParams = None, device="cuda",
                    window_size_samples=None,
                    progress_tracking_callback=None):
    """使用Silero VAD进行语音活动检测

    Args:
        audio_path: 音频文件路径
        vad_threshold: VAD阈值 (0.0-1.0)
        device: 计算设备
        min_speech_duration: 最小语音片段时长（秒），小于此值的片段会被过滤
        min_silence_duration: 静音等待时长（秒），用于分割语音片段
        speech_pad_ms: 语音填充（毫秒），在每个语音片段前后添加
        prefix_padding_ms: 前缀填充（毫秒），在每个语音片段开头添加
        max_speech_duration_s: 最大语音片段时长（秒），超过此值会强制分割
        neg_threshold: 静音判定阈值，默认为 threshold-0.15
        min_silence_at_max_speech: 长语音分割时最大片段内静音时长（毫秒）
        use_max_poss_sil_at_max_speech: 是否优先使用最长静音分割点
        window_size_samples: 窗口大小（样本数）
        time_resolution: 时间分辨率
        progress_tracking_callback: 进度回调函数

    Returns:
        包含语音片段时间戳的列表 [(start, end), ...]
    """
    if vad_params is None:
        vad_params = VadParams()
    vad_threshold = vad_params.threshold
    min_speech_duration = vad_params.min_speech
    min_silence_duration = vad_params.min_silence
    speech_pad_ms = vad_params.speech_pad_ms
    prefix_padding_ms = vad_params.prefix_padding_ms
    max_speech_duration_s = vad_params.max_speech
    neg_threshold = vad_params.neg_threshold
    min_silence_at_max_speech = vad_params.max_sil_dur
    use_max_poss_sil_at_max_speech = vad_params.max_sil_split
    time_resolution = vad_params.resolution

    if not SILERO_VAD_AVAILABLE:
        timestamp_print("[VAD] Silero VAD 不可用，跳过VAD处理")
        return []

    try:
        from silero_vad import get_speech_timestamps as get_sts, read_audio as read_wav

        if not hasattr(apply_silero_vad, '_cached_model') or apply_silero_vad._cached_model is None:
            local_model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'silero_vad')
            local_model_path = os.path.join(local_model_dir, 'silero_vad.jit')

            if os.path.exists(local_model_path):
                print(f"[VAD] 从本地加载 Silero VAD: {local_model_path}")
                model = torch.jit.load(local_model_path)
            else:
                print("[VAD] 从 torch.hub 加载 Silero VAD...")
                model, utils = torch.hub.load(
                    repo_or_dir='snakers4/silero-vad',
                    model='silero_vad',
                    verbose=False
                )
                get_sts, _, read_wav, _, _ = utils
            apply_silero_vad._cached_model = model
        else:
            model = apply_silero_vad._cached_model

        model = model.to(device)
        model.eval()
        wav = read_wav(audio_path, sampling_rate=16000)
        if wav.shape[0] == 2:
            wav = wav.mean(dim=1)
        wav = wav.to(device)

        # 构建 get_speech_timestamps 参数
        vad_kwargs = {
            'threshold': vad_threshold,
            'min_speech_duration_ms': int(min_speech_duration * 1000),
            'min_silence_duration_ms': int(min_silence_duration * 1000),
            'speech_pad_ms': speech_pad_ms,
            'max_speech_duration_s': max_speech_duration_s if max_speech_duration_s != float('inf') else float('inf'),
            'sampling_rate': 16000,
            'return_seconds': True,
        }

        # 添加可选参数
        if neg_threshold is not None:
            vad_kwargs['neg_threshold'] = neg_threshold
        if min_silence_at_max_speech is not None:
            vad_kwargs['min_silence_at_max_speech'] = min_silence_at_max_speech
        if use_max_poss_sil_at_max_speech is not None:
            vad_kwargs['use_max_poss_sil_at_max_speech'] = use_max_poss_sil_at_max_speech
        if window_size_samples is not None:
            vad_kwargs['window_size_samples'] = window_size_samples
        if time_resolution is not None:
            vad_kwargs['time_resolution'] = time_resolution
        if progress_tracking_callback is not None:
            vad_kwargs['progress_tracking_callback'] = progress_tracking_callback
        speech_timestamps = get_sts(wav, model=model, **vad_kwargs)

        timestamp_print(f"[VAD] 检测参数: threshold={vad_threshold}, min_speech={min_speech_duration}s, min_silence={min_silence_duration}s, speech_pad={speech_pad_ms}ms, prefix_pad={prefix_padding_ms}ms")
        timestamp_print(f"[VAD] 原始语音片段数: {len(speech_timestamps)}")

        speech_segments = []
        for idx, ts in enumerate(speech_timestamps):
            start = max(0, ts['start'] - prefix_padding_ms / 1000)
            end = ts['end']
            duration = end - start
            timestamp_print(f"[VAD] 片段 {idx+1}: {start:.2f}s - {end:.2f}s (时长: {duration:.2f}s)")
            speech_segments.append((start, end))

        timestamp_print(f"[VAD] 总语音时长: {sum(e-s for s,e in speech_segments):.2f}s")
        timestamp_print(f"[VAD] 总片段数: {len(speech_segments)}")

        del wav
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()

        return speech_segments

    except Exception as e:
        timestamp_print(f"[VAD] Silero VAD 处理失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return []

def recognize_speech_enhanced(audio_path, model_path, detected_language=None, device_choice="auto",
                    progress_callback=None, word_timestamps=True,
                    vad_params: VadParams = None,
                    cd_params: CdParams = None,
                    enable_alignment=True,
                    enable_vad=True):
    """增强版语音识别，支持 VAD 处理

    Args:
        audio_path: 音频文件路径
        model_path: Whisper模型名称或路径
        detected_language: 检测到的语言
        device_choice: 设备选择
        progress_callback: 进度回调
        word_timestamps: 是否启用单词时间戳
        vad_threshold: VAD阈值 (0.0-1.0)
        min_speech_duration: 最小语音片段时长（秒）
        max_speech_duration: 最大语音片段时长（秒），超过此长度会强制分割
        min_silence_duration: 最小静音间隔（秒）
        speech_pad_ms: 语音填充（毫秒）
        prefix_padding_ms: 前缀填充（毫秒）
        use_max_poss_sil_at_max_speech: 是否优先使用最长静音分割点
        enable_alignment: 是否启用强制对齐
        enable_vad: 是否启用VAD处理
        neg_threshold: VAD 静音判定阈值，默认为 threshold-0.15

    Returns:
        识别结果字典
    """
    timestamp_print("[语音识别] 使用增强版语音识别器进行语音识别")

    if vad_params is None:
        vad_params = VadParams()
    if cd_params is None:
        cd_params = CdParams()

    device = "cuda" if device_choice != "cpu" else "cpu"
    speech_segments = []

    if enable_vad:
        speech_segments = apply_silero_vad(
            audio_path,
            vad_params=vad_params,
            device=device
        )
        timestamp_print(f"[VAD] 检测到 {len(speech_segments)} 个语音片段")

    from config import config
    from utils.whisper_cd_original import WhisperCDOriginal

    timestamp_print("[Whisper-CD] 启用 Whisper-CD 处理器...")

    whispercd_processor = WhisperCDOriginal(
        model_path=model_path,
        cd_params=cd_params,
        enable_alignment=enable_alignment,
        speech_segments=speech_segments if enable_vad else None
    )

    timestamp_print("[Whisper-CD] 应用对比解码...")
    cd_result = whispercd_processor.transcribe(
        audio_path,
        detected_language,
        progress_callback=progress_callback
    )

    result = _process_cd_segments(cd_result, speech_segments if enable_vad else None)

    timestamp_print(f"[VAD] 最终保留 {len(result['segments'])} 个识别片段")

    whispercd_processor.cleanup()
    del whispercd_processor
    _cleanup_memory()

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        timestamp_print(f"[内存管理] Whisper-CD 卸载后 GPU 状态: 已分配 {allocated:.2f}GB, 已保留 {reserved:.2f}GB")

    if progress_callback:
        progress_callback(80)

    if enable_alignment and result.get('segments'):
        aligner = None
        try:
            timestamp_print("[强制对齐] 启用强制对齐...")
            aligner = ForcedAligner(device=device)
            if aligner.load_alignment_model(detected_language or result.get('language', 'ja')):
                result['segments'] = aligner.align(result['segments'], audio_path, return_char_alignments=True)
                timestamp_print("[强制对齐] 强制对齐完成")
            else:
                timestamp_print("[强制对齐] 强制对齐模型加载失败，跳过对齐")
        except Exception as e:
            timestamp_print(f"[强制对齐] 强制对齐失败: {str(e)}")
        finally:
            if aligner is not None:
                aligner.cleanup()
                timestamp_print("[内存管理] 强制对齐模型已卸载，清理显存")
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    reserved = torch.cuda.memory_reserved() / 1024**3
                    timestamp_print(f"[内存管理] 强制对齐卸载后 GPU 状态: 已分配 {allocated:.2f}GB, 已保留 {reserved:.2f}GB")

    return result


def clear_model_cache():
    """清理模型缓存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    timestamp_print("[缓存] 模型缓存已清理")
