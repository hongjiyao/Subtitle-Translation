# -*- coding: utf-8 -*-
"""
语音识别模块
包含基础语音识别和强制对齐功能

"""

import os
import datetime
import gc
import subprocess
import tempfile
import re
import torch
import numpy as np
import torchaudio
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from config import MODEL_CACHE_DIR

try:
    from silero_vad import get_speech_timestamps, read_audio
    SILERO_VAD_AVAILABLE = True
except ImportError:
    SILERO_VAD_AVAILABLE = False


# 公共工具函数
def timestamp_print(message):
    """带时间戳的打印函数"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


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

    def _detect_text_language(self, text: str) -> str:
        """自动检测文本的主要语言"""
        import re
        japanese_chars = re.findall(r'[\u3040-\u309f\u30a0-\u30ff]', text)
        chinese_chars = re.findall(r'[\u4e00-\u9faf]', text)
        korean_chars = re.findall(r'[\ac00-\ud7af]', text)

        if len(japanese_chars) > 0 or len(chinese_chars) > 0:
            return 'ja'
        elif len(korean_chars) > 0:
            return 'ko'
        return 'en'

    def _preprocess_text_for_alignment(self, text: str, language: str = 'en') -> str:
        """
        文本预处理以适配 Wav2Vec2 强制对齐

        功能：
        1. 去除 HTML 标签
        2. 去除 URL
        3. 数字转为文字
        4. 过滤所有标点符号

        Args:
            text: 原始文本
            language: 语言代码（未使用，仅保持接口兼容）

        Returns:
            预处理后的文本
        """
        import re

        if not text:
            return ""

        # 1. 去除 HTML 标签
        text = re.sub(r'<[^>]+>', '', text)

        # 2. 去除 URL
        text = re.sub(r'http[s]?://\S+', '', text)

        # 3. 数字转为文字（统一处理，不区分语言）
        def replace_number(match):
            number_str = match.group(0)
            try:
                number = int(number_str)
                return self._number_to_words(number, 'en')
            except:
                return number_str

        text = re.sub(r'\b\d+\b', replace_number, text)

        # 4. 过滤所有标点符号，保留字母、数字和空格
        # 包括：日文标点、英文标点、省略号等
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
        timestamp_print(f"[强制对齐] 使用设备: {self.device}")

        try:
            aligned_segments = []
            sr = 16000  # Wav2Vec2 模型使用 16kHz 采样率

            timestamp_print(f"[强制对齐] 开始处理 {len(transcript_segments)} 个段落")

            for i, segment in enumerate(transcript_segments):
                text = segment.get('text', '')
                start_time = segment.get('start', 0.0)
                end_time = segment.get('end', 0.0)

                # 获取语言代码用于文本预处理
                language = segment.get('language', 'en')

                timestamp_print(f"[强制对齐] 处理段落 {i+1}/{len(transcript_segments)}")
                timestamp_print(f"[强制对齐] 时间范围: {start_time:.2f}s - {end_time:.2f}s")
                timestamp_print(f"[强制对齐] 原始文本: '{text[:100]}...'" if len(text) > 100 else f"[强制对齐] 原始文本: '{text}'")

                # 预处理文本以适配 Wav2Vec2 tokenizer
                original_text = text
                text = self._preprocess_text_for_alignment(text, language)

                timestamp_print(f"[强制对齐] 预处理后文本: '{text[:100]}...'" if len(text) > 100 else f"[强制对齐] 预处理后文本: '{text}'")

                if not text:
                    timestamp_print(f"[强制对齐] 跳过空文本段落")
                    aligned_segments.append(segment)
                    continue

                # 使用 ffmpeg 提取音频片段
                timestamp_print(f"[强制对齐] 提取音频片段...")
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    temp_audio_path = temp_file.name

                try:
                    # 使用 ffmpeg 提取指定时间段的音频
                    duration = end_time - start_time
                    timestamp_print(f"[强制对齐] 音频时长: {duration:.2f}s")
                    subprocess.run([
                        'ffmpeg', '-y', '-i', audio_path,
                        '-ss', str(start_time), '-t', str(duration),
                        '-ar', str(sr), '-ac', '1', '-acodec', 'pcm_s16le',
                        temp_audio_path
                    ], check=True, capture_output=True)
                    timestamp_print(f"[强制对齐] 音频提取成功")

                    # 使用 torchaudio 加载音频
                    waveform, sample_rate = torchaudio.load(temp_audio_path)
                    timestamp_print(f"[强制对齐] 音频加载成功，采样率: {sample_rate}")

                    # 确保是单声道
                    if waveform.shape[0] > 1:
                        timestamp_print(f"[强制对齐] 转换为单声道")
                        waveform = torch.mean(waveform, dim=0, keepdim=True)

                    # 转换为 numpy 数组
                    segment_audio = waveform.squeeze().numpy()
                    timestamp_print(f"[强制对齐] 音频数据形状: {segment_audio.shape}")

                except Exception as e:
                    timestamp_print(f"[强制对齐警告] 提取音频片段失败: {str(e)}")
                    aligned_segments.append(segment)
                    continue
                finally:
                    # 清理临时文件
                    try:
                        os.unlink(temp_audio_path)
                        timestamp_print(f"[强制对齐] 清理临时音频文件")
                    except:
                        pass

                if len(segment_audio) == 0:
                    timestamp_print(f"[强制对齐] 音频片段为空，跳过")
                    aligned_segments.append(segment)
                    continue

                # 提取音频特征进行对齐
                timestamp_print(f"[强制对齐] 提取音频特征...")
                inputs = self.align_processor(segment_audio, sampling_rate=sr, return_tensors="pt")
                input_values = inputs.input_values.to(self.device)
                timestamp_print(f"[强制对齐] 输入形状: {input_values.shape}")

                # 如果音频太长，分块处理以避免 GPU 显存不足
                max_length = 16000 * 30  # 最大30秒
                if input_values.shape[1] > max_length:
                    # 分段处理
                    num_chunks = (input_values.shape[1] + max_length - 1) // max_length
                    timestamp_print(f"[强制对齐] 音频过长，分 {num_chunks} 块处理")
                    logits = []
                    for chunk_idx in range(num_chunks):
                        start_idx = chunk_idx * max_length
                        end_idx = min((chunk_idx + 1) * max_length, input_values.shape[1])
                        chunk_input = input_values[:, start_idx:end_idx]
                        timestamp_print(f"[强制对齐] 处理块 {chunk_idx+1}/{num_chunks}")

                        with torch.no_grad():
                            chunk_logits = self.align_model(chunk_input).logits
                            logits.append(chunk_logits)
                        timestamp_print(f"[强制对齐] 块处理完成，logits形状: {chunk_logits.shape}")

                        # 清理显存
                        torch.cuda.empty_cache()

                    # 合并 logits
                    logits = torch.cat(logits, dim=1)
                    timestamp_print(f"[强制对齐] 合并后logits形状: {logits.shape}")
                else:
                    # 检查输入长度
                    if input_values.shape[1] < 100:  # 最小输入长度
                        # 输入太短，使用空 logits
                        logits = torch.zeros((1, 1, self.align_model.config.vocab_size), device=self.device)
                        timestamp_print(f"[强制对齐] 输入太短，使用空 logits")
                    else:
                        # 推断
                        with torch.no_grad():
                            logits = self.align_model(input_values).logits
                        timestamp_print(f"[强制对齐] 模型推理完成，logits形状: {logits.shape}")

                # 清理显存
                torch.cuda.empty_cache()

                # 准备标签 - 需要是二维张量 (batch_size, seq_len)
                timestamp_print(f"[强制对齐] 准备标签...")
                tokens = self.align_processor.tokenizer.tokenize(text)
                timestamp_print(f"[强制对齐] Token数量: {len(tokens)}")
                if not tokens:
                    timestamp_print(f"[强制对齐] Tokenizer返回空tokens")
                    aligned_segments.append(segment)
                    continue
                labels = [self.align_processor.tokenizer.convert_tokens_to_ids(token) for token in tokens]
                labels = torch.tensor([labels], device=self.device)  # 添加 batch 维度
                timestamp_print(f"[强制对齐] Labels形状: {labels.shape}")

                if labels.shape[1] == 0:
                    timestamp_print(f"[强制对齐] Labels为空")
                    aligned_segments.append(segment)
                    continue

                # 获取对齐路径和分数（用于获取时间步信息）
                input_length = logits.shape[1]
                target_length = labels.shape[1]
                max_target_length = input_length // 2
                timestamp_print(f"[强制对齐] 输入长度: {input_length}, 目标长度: {target_length}")

                if target_length > max_target_length:
                    # 分割段落
                    split_size = max_target_length - 10  # 留一些余量
                    splits = []
                    for i in range(0, target_length, split_size):
                        end = min(i + split_size, target_length)
                        splits.append((i, end))
                    timestamp_print(f"[强制对齐] 段落过长，分 {len(splits)} 段处理")

                    # 对每个子段落执行对齐
                    split_paths = []
                    split_scores = []

                    for start, end in splits:
                        split_labels = labels[:, start:end]
                        split_target_lengths = torch.tensor([end - start], device=self.device)
                        timestamp_print(f"[强制对齐] 处理子段落 {start}-{end}")

                        from torchaudio.functional import forced_align
                        try:
                            split_path, split_score = forced_align(
                                logits,
                                split_labels,
                                input_lengths=torch.tensor([input_length], device=self.device),
                                target_lengths=split_target_lengths,
                                blank=self.align_processor.tokenizer.pad_token_id
                            )
                            split_paths.extend(split_path[0].tolist())
                            split_scores.extend(split_score[0].tolist())
                            timestamp_print(f"[强制对齐] 子段落对齐成功")
                        except Exception as e:
                            # 如果分割后仍然失败，使用空结果
                            split_paths.extend([-1] * (end - start))
                            split_scores.extend([0.0] * (end - start))
                            timestamp_print(f"[强制对齐] 子段落对齐失败: {str(e)}")

                    paths = torch.tensor([split_paths], device=self.device)
                    scores = torch.tensor([split_scores], device=self.device)
                    timestamp_print(f"[强制对齐] 合并后路径形状: {paths.shape}")
                else:
                    # 正常执行强制对齐
                    input_lengths = torch.tensor([input_length], device=self.device)  # (B,)
                    target_lengths = torch.tensor([target_length], device=self.device)  # (B,)
                    timestamp_print(f"[强制对齐] 执行强制对齐...")

                    from torchaudio.functional import forced_align
                    paths, scores = forced_align(
                        logits,
                        labels,
                        input_lengths=input_lengths,
                        target_lengths=target_lengths,
                        blank=self.align_processor.tokenizer.pad_token_id
                    )
                    timestamp_print(f"[强制对齐] 对齐完成，路径形状: {paths.shape}")

                # 计算时间戳
                frame_duration = (end_time - start_time) / logits.shape[1]
                timestamp_print(f"[强制对齐] 帧持续时间: {frame_duration:.6f}s")

                # 收集每个token位置对应的logits（用于CD计算）
                if return_char_alignments:
                    timestamp_print(f"[强制对齐] 生成字符级对齐...")
                    chars = []

                    # 遍历对齐路径中的每个时间步
                    for t in range(paths.shape[1]):
                        token_id = paths[0, t].item()
                        token = self.align_processor.tokenizer.convert_ids_to_tokens([token_id])[0]

                        if token not in ['[PAD]', '[UNK]', '|', '<pad>', '<unk>']:
                            # 处理特殊标记
                            token = token.replace('▁', ' ').strip()
                            if token:
                                for char in token:
                                    if char not in ['[PAD]', '[UNK]', '|', '<pad>', '<unk>']:
                                        char_start = start_time + t * frame_duration
                                        char_end = start_time + (t + 1) * frame_duration

                                        # 不收集logits分数，减小文件大小
                                        chars.append({
                                            'char': char,
                                            'start': char_start,
                                            'end': char_end
                                        })
                    segment['chars'] = chars
                    timestamp_print(f"[强制对齐] 字符级对齐完成，字符数: {len(chars)}")

                # 将 chars 转换为 words（用于字幕生成）
                if 'chars' in segment and segment['chars']:
                    timestamp_print(f"[强制对齐] 生成单词级对齐...")
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

                # 保留原始文本，只更新时间戳
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


class EnhancedSpeechRecognizer:
    """增强版语音识别器 - 集成对齐、断句和单词级时间戳"""
    
    def __init__(self, 
                 model_path: str, 
                 device: str = "cuda", 
                 compute_type: str = "float16", 
                 enable_alignment: bool = True):
        """
        初始化增强版语音识别器
        
        Args:
            model_path: Whisper 模型路径
            device: 计算设备
            compute_type: 计算类型
            enable_alignment: 是否启用强制对齐
        """
        self.model_path = model_path
        self.device = device
        self.compute_type = compute_type
        self.enable_alignment = enable_alignment
        
        # 初始化组件
        self.whisper_model = None
        self.whisper_processor = None
        self.aligner = None
        
        if self.enable_alignment:
            self.aligner = ForcedAligner(device, compute_type)
    
    def load_model(self):
        """加载 Whisper 模型"""
        timestamp_print(f"[模型加载] 正在加载 Whisper 模型...")
        
        # 只使用原始Whisper模型
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        
        self.whisper_model = WhisperForConditionalGeneration.from_pretrained(self.model_path)
        self.whisper_processor = WhisperProcessor.from_pretrained(self.model_path)
        self.whisper_model.to(self.device)
        timestamp_print(f"[模型加载] 原始Whisper模型加载完成")
    
    def transcribe(self, 
                   audio_path: str, 
                   language: str = None, 
                   word_timestamps: bool = True) -> dict:
        """
        执行增强版语音识别
        
        Args:
            audio_path: 音频文件路径
            language: 语言代码（None 表示自动检测）
            word_timestamps: 是否生成单词级时间戳
            
        Returns:
            包含对齐、断句和单词级时间戳的完整结果
        """
        if self.whisper_model is None:
            self.load_model()
        
        # 步骤 1: 基础语音识别
        timestamp_print(f"[语音识别] 执行基础识别...")
        
        # 只使用原始Whisper模型
        import librosa
        
        # 加载音频
        audio_data, _ = librosa.load(audio_path, sr=16000)
        
        # 预处理
        inputs = self.whisper_processor(audio_data, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device) if 'attention_mask' in inputs else None
        
        # 生成
        with torch.no_grad():
            generate_kwargs = {
                "input_features": input_features,
                "language": language,
                "task": "transcribe"
            }
            if attention_mask is not None:
                generate_kwargs["attention_mask"] = attention_mask

            outputs = self.whisper_model.generate(**generate_kwargs)
        
        # 解码
        transcription = self.whisper_processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0]
        
        # 构建结果
        result = {
            'segments': [{'text': transcription, 'start': 0, 'end': len(audio_data)/16000}],
            'text': transcription,
            'language': language or 'en'
        }
        
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
                    return_char_alignments=True
                )
            else:
                # 对齐模型加载失败，使用原始段
                aligned_segments = raw_segments
                self.enable_alignment = False  # 标记对齐未成功
        else:
            aligned_segments = raw_segments
        
        # 直接使用原始结果，不进行自定义断句
        final_segments = aligned_segments if aligned_segments else []
        
        # 构建最终结果
        full_text = ''.join(seg['text'] for seg in final_segments)
        
        result_data = {
            'text': full_text,
            'language': detected_language,
            'segments': final_segments,
            'word_timestamps_enabled': word_timestamps,
            'alignment_enabled': self.enable_alignment,
            'segmentation_enabled': True  # 始终启用断句
        }
        
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
        
        # 添加语言信息到每个segment
        segment_dict = {
            'start': start,
            'end': end,
            'text': text,
            'words': words,
            'chars': chars,
            'language': cd_result.get('language', 'en')
        }

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


def recognize_speech(audio_path, model_path, detected_language=None, device_choice="auto",
                    progress_callback=None, enable_whispercd=None,
                    word_timestamps=True, enable_vad=True):
    """使用增强版语音识别器进行语音识别"""
    return recognize_speech_enhanced(
        audio_path=audio_path,
        model_path=model_path,
        detected_language=detected_language,
        device_choice=device_choice,
        progress_callback=progress_callback,
        word_timestamps=word_timestamps,
        enable_vad=enable_vad,
        enable_whispercd=enable_whispercd
    )


def apply_silero_vad(audio_path, vad_threshold=0.4, device="cuda",
                    min_speech_duration=1.0, min_silence_duration=1.0,
                    speech_pad_ms=300, prefix_padding_ms=50,
                    max_speech_duration_s=float('inf'),
                    neg_threshold=None,
                    min_silence_at_max_speech=98,
                    use_max_poss_sil_at_max_speech=False,
                    window_size_samples=None,
                    time_resolution=64,
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
    if not SILERO_VAD_AVAILABLE:
        timestamp_print("[VAD] Silero VAD 不可用，跳过VAD处理")
        return []

    try:
        # 获取 get_speech_timestamps 和 read_audio 函数
        from silero_vad import get_speech_timestamps as get_sts, read_audio as read_wav

        # 首先尝试从本地目录加载 Silero VAD 模型
        local_model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'silero_vad')
        local_model_path = os.path.join(local_model_dir, 'silero_vad.jit')

        if os.path.exists(local_model_path):
            print(f"[VAD] 从本地加载 Silero VAD: {local_model_path}")
            model = torch.jit.load(local_model_path)
        else:
            # 从 torch.hub 加载（需要网络）
            print("[VAD] 从 torch.hub 加载 Silero VAD...")
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                verbose=False
            )
            get_sts, _, read_wav, _, _ = utils

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

        del model
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
                    vad_threshold=0.4, min_speech_duration=1.0, max_speech_duration=30.0, min_silence_duration=1.0,
                    speech_pad_ms=300, prefix_padding_ms=50,
                    use_max_poss_sil_at_max_speech=True,
                    enable_alignment=True, enable_whispercd=True,
                    enable_vad=True, enable_punctuate=True,
                    neg_threshold=None):
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
        enable_whispercd: 是否启用Whisper-CD
        enable_vad: 是否启用VAD处理
        enable_punctuate: 是否启用标点还原
        neg_threshold: VAD 静音判定阈值，默认为 threshold-0.15

    Returns:
        识别结果字典
    """
    timestamp_print("[语音识别] 使用增强版语音识别器进行语音识别")

    device = "cuda" if device_choice != "cpu" else "cpu"
    speech_segments = []

    if enable_vad:
        speech_segments = apply_silero_vad(
            audio_path,
            vad_threshold=vad_threshold,
            device=device,
            min_speech_duration=min_speech_duration,
            max_speech_duration_s=max_speech_duration,
            min_silence_duration=min_silence_duration,
            speech_pad_ms=speech_pad_ms,
            prefix_padding_ms=prefix_padding_ms,
            use_max_poss_sil_at_max_speech=use_max_poss_sil_at_max_speech,
            neg_threshold=neg_threshold
        )
        timestamp_print(f"[VAD] 检测到 {len(speech_segments)} 个语音片段")

    if enable_whispercd:
        from config import config
        from utils.whisper_cd_original import WhisperCDOriginal

        timestamp_print("[Whisper-CD] 启用 Whisper-CD 处理器...")

        whispercd_processor = WhisperCDOriginal(
            model_path=model_path,
            alpha=config.get("whispercd_alpha", 1.0),
            temperature=config.get("whispercd_temperature", 1.0),
            snr_db=config.get("whispercd_snr_db", 10),
            temporal_shift=config.get("whispercd_temporal_shift", 7),
            score_threshold=config.get("whispercd_score_threshold", 0.3),
            batch_size=config.get("whispercd_batch_size", 4),
            enable_alignment=enable_alignment,
            speech_segments=speech_segments if enable_vad else None,
            context_segments=config.get("whispercd_context_segments", 10)
        )

        timestamp_print("[Whisper-CD] 应用对比解码...")
        cd_result = whispercd_processor.transcribe(
            audio_path,
            detected_language,
            progress_callback=progress_callback
        )

        result = _process_cd_segments(cd_result, speech_segments if enable_vad else None)

        timestamp_print(f"[VAD] 最终保留 {len(result['segments'])} 个识别片段")

        timestamp_print("[内存管理] Whisper-CD 处理完成，清理内存和显存...")
        _cleanup_memory()

        if progress_callback:
            progress_callback(80)

        if enable_alignment and result.get('segments'):
            try:
                from utils.whisper_cd_original import WhisperCDOriginal
                timestamp_print("[强制对齐] 启用强制对齐...")
                aligner = ForcedAligner(device=device)
                if aligner.load_alignment_model(detected_language or result.get('language', 'ja')):
                    result['segments'] = aligner.align(result['segments'], audio_path, return_char_alignments=True)
                    timestamp_print("[强制对齐] 强制对齐完成")
                    aligner.cleanup()
                    timestamp_print("[内存管理] 强制对齐模型已卸载，清理显存")
                else:
                    timestamp_print("[强制对齐] 强制对齐模型加载失败，跳过对齐")
            except Exception as e:
                timestamp_print(f"[强制对齐] 强制对齐失败: {str(e)}")

        # 移除标点处理功能
        if enable_punctuate and result.get('segments'):
            timestamp_print("[标点处理] 标点处理功能已移除")

    return result


def clear_model_cache():
    """清理模型缓存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    timestamp_print("[缓存] 模型缓存已清理")
