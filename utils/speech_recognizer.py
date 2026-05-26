# -*- coding: utf-8 -*-
"""
语音识别模块
包含基础语音识别和强制对齐功能

"""

import os
import gc
import re
import torch
import torchaudio
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from config import PROJECT_ROOT

from config import MODEL_CACHE_DIR, CdParams



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
                except Exception:
                    print(f"[模型检查] 列出快照目录失败: {path}")
                    pass
            elif os.path.isdir(path):
                return path
    
    return None


class ForcedAligner:
    """强制对齐模块 - 使用 torchaudio 和 Wav2Vec2 进行帧级对齐"""
    
    def __init__(self, device: str = "auto"):
        """
        初始化强制对齐器
        
        Args:
            device: 计算设备 ("auto", "cuda" 或 "cpu")
        """
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
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
        print(f"[强制对齐] 正在加载对齐模型 (语言: {language_code})...")
        
        # 检查本地是否存在wav2vec2模型
        wav2vec2_models = {
            "ar": "jonatasgrosman--wav2vec2-large-xlsr-53-arabic",
            "nl": "jonatasgrosman--wav2vec2-large-xlsr-53-dutch",
            "en": "jonatasgrosman--wav2vec2-large-xlsr-53-english",
            "fi": "jonatasgrosman--wav2vec2-large-xlsr-53-finnish",
            "fr": "jonatasgrosman--wav2vec2-large-xlsr-53-french",
            "de": "jonatasgrosman--wav2vec2-large-xlsr-53-german",
            "it": "jonatasgrosman--wav2vec2-large-xlsr-53-italian",
            "ja": "jonatasgrosman--wav2vec2-large-xlsr-53-japanese",
            "fa": "jonatasgrosman--wav2vec2-large-xlsr-53-persian",
            "pl": "jonatasgrosman--wav2vec2-large-xlsr-53-polish",
            "pt": "jonatasgrosman--wav2vec2-large-xlsr-53-portuguese",
            "ru": "jonatasgrosman--wav2vec2-large-xlsr-53-russian",
            "zh": "jonatasgrosman--wav2vec2-large-xlsr-53-chinese-zh-cn",
            "es": "jonatasgrosman--wav2vec2-large-xlsr-53-spanish",
        }
        
        model_exists = False
        model_path = None
        if language_code in wav2vec2_models:
            model_name = wav2vec2_models[language_code]
            model_path = os.path.join(MODEL_CACHE_DIR, model_name)
            if os.path.exists(model_path):
                print(f"[强制对齐] 找到本地wav2vec2模型: {model_path}")
                model_exists = True
            else:
                print(f"[强制对齐警告] 本地wav2vec2模型不存在: {model_path}")
        else:
            print(f"[强制对齐警告] 语言 {language_code} 没有对应的wav2vec2模型")
        
        try:
            if model_exists:
                # 检查模型文件是否存在
                required_files = ['config.json', 'preprocessor_config.json']
                missing_files = [f for f in required_files if not os.path.exists(os.path.join(model_path, f))]

                # pytorch_model.bin 和 model.safetensors 至少存在一个
                model_weight_files = ['pytorch_model.bin', 'model.safetensors']
                if not any(os.path.exists(os.path.join(model_path, f)) for f in model_weight_files):
                    missing_files.append(f"({' 或 '.join(model_weight_files)})")

                # vocab.json 和 tokenizer.json 至少存在一个
                vocab_files = ['vocab.json', 'tokenizer.json']
                if not any(os.path.exists(os.path.join(model_path, f)) for f in vocab_files):
                    missing_files.append(f"({' 或 '.join(vocab_files)})")

                if missing_files:
                    print(f"[强制对齐警告] 缺少必要的模型文件: {missing_files}")
                    return False
                
                # 使用 torchaudio 和 transformers 加载模型
                from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
                
                # 加载处理器和模型
                self.align_processor = Wav2Vec2Processor.from_pretrained(model_path)
                dtype = torch.float16 if self.device == "cuda" else torch.float32
                self.align_model = Wav2Vec2ForCTC.from_pretrained(model_path, torch_dtype=dtype)
                self.align_model.to(self.device)
                self.align_model.eval()
                
                print(f"[强制对齐] 对齐模型加载完成")
                return True
            else:
                print(f"[强制对齐警告] 本地模型不存在，跳过强制对齐")
                self.align_model = None
                self.align_processor = None
                return False

        except Exception as e:
            print(f"[强制对齐警告] 无法加载对齐模型: {str(e)}")
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
                except Exception:
                    print(f"[强制对齐] 数字转文字失败: {number_str}")
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

        print(f"[强制对齐] 开始帧级对齐处理...")

        try:
            aligned_segments = []
            sr = 16000

            full_waveform, sample_rate = torchaudio.load(audio_path)
            if full_waveform.shape[0] > 1:
                full_waveform = torch.mean(full_waveform, dim=0, keepdim=True)
            if sample_rate != sr:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=sr)
                full_waveform = resampler(full_waveform)
            full_audio = full_waveform.squeeze().numpy()

            print(f"[强制对齐] 开始处理 {len(transcript_segments)} 个段落")

            for i, segment in enumerate(transcript_segments):
                text = segment.get('text', '')
                start_time = segment.get('start', 0.0)
                end_time = segment.get('end', 0.0)

                language = segment.get('language', 'en')

                print(f"[强制对齐] 处理段落 {i+1}/{len(transcript_segments)}")

                original_text = text
                text = self._preprocess_text_for_alignment(text, language)

                if not text:
                    print(f"[强制对齐] 跳过空文本段落")
                    aligned_segments.append(segment)
                    continue

                try:
                    start_sample = int(start_time * sr)
                    end_sample = int(end_time * sr)
                    segment_audio = full_audio[start_sample:end_sample]
                except Exception as e:
                    print(f"[强制对齐警告] 提取音频片段失败: {str(e)}")
                    aligned_segments.append(segment)
                    continue

                if len(segment_audio) == 0:
                    print(f"[强制对齐] 音频片段为空，跳过")
                    aligned_segments.append(segment)
                    continue

                inputs = self.align_processor(segment_audio, sampling_rate=sr, return_tensors="pt")
                input_values = inputs.input_values.to(self.device)
                model_dtype = next(self.align_model.parameters()).dtype
                if input_values.dtype != model_dtype:
                    input_values = input_values.to(model_dtype)

                max_length = 16000 * 30
                if input_values.shape[1] > max_length:
                    num_chunks = (input_values.shape[1] + max_length - 1) // max_length
                    print(f"[强制对齐] 音频过长，分 {num_chunks} 块处理")
                    logits = []
                    for chunk_idx in range(num_chunks):
                        start_idx = chunk_idx * max_length
                        end_idx = min((chunk_idx + 1) * max_length, input_values.shape[1])
                        chunk_input = input_values[:, start_idx:end_idx]

                        with torch.no_grad():
                            chunk_logits = self.align_model(chunk_input).logits
                            logits.append(chunk_logits)

                    logits = torch.cat(logits, dim=1)
                else:
                    if input_values.shape[1] < 100:
                        model_dtype = next(self.align_model.parameters()).dtype
                        logits = torch.zeros((1, 1, self.align_model.config.vocab_size), device=self.device, dtype=model_dtype)
                        print(f"[强制对齐] 输入太短，使用空 logits")
                    else:
                        with torch.no_grad():
                            logits = self.align_model(input_values).logits

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

                if input_length < 2:
                    aligned_segments.append(segment)
                    continue

                max_target_length = input_length // 2

                if target_length > max_target_length:
                    split_size = max(1, max_target_length - 10)
                    if split_size < 1:
                        split_size = 1
                    all_token_alignments = []
                    print(f"[强制对齐] 段落过长，分 {(target_length + split_size - 1) // split_size} 段处理")
                    
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
                        
                        if logits_end <= logits_start:
                            logits_end = min(input_length, logits_start + 1)
                        
                        split_logits = logits[:, logits_start:logits_end]
                        split_input_length = split_logits.shape[1]
                        
                        if split_input_length < split_target_length * 2:
                            continue
                        
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
                            print(f"[强制对齐] 子段对齐失败: {str(e)}")
                    
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
                    merged_alignments = []
                    for t, token in token_alignments:
                        token = token.replace('▁', ' ').strip()
                        if not token:
                            continue
                        for char in token:
                            if char in ['[PAD]', '[UNK]', '|', '<pad>', '<unk>']:
                                continue
                            char_start = start_time + t * frame_duration
                            char_end = start_time + (t + 1) * frame_duration
                            if (merged_alignments and 
                                merged_alignments[-1]['char'] == char and
                                char_start - merged_alignments[-1]['end'] < frame_duration * 2):
                                merged_alignments[-1]['end'] = char_end
                            else:
                                merged_alignments.append({
                                    'char': char,
                                    'start': char_start,
                                    'end': char_end
                                })
                    segment['chars'] = merged_alignments
                    print(f"[强制对齐] 字符级对齐完成，字符数: {len(merged_alignments)}")

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
                    print(f"[强制对齐] 单词级对齐完成，单词数: {len(words)}")

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

                # 根据 chars/words 修正 segment 的 start/end 时间戳
                original_start = segment.get('start')
                original_end = segment.get('end')

                if 'chars' in segment and segment['chars']:
                    valid_chars = [c for c in segment['chars']
                                   if c.get('start', 0) >= 0 and c.get('end', 0) > c.get('start', 0)]
                    if valid_chars:
                        segment['start'] = valid_chars[0]['start']
                        segment['end'] = valid_chars[-1]['end']
                elif 'words' in segment and segment['words']:
                    valid_words = [w for w in segment['words']
                                   if w.get('start', 0) >= 0 and w.get('end', 0) > w.get('start', 0)]
                    if valid_words:
                        segment['start'] = valid_words[0]['start']
                        segment['end'] = valid_words[-1]['end']

                if segment.get('start') != original_start or segment.get('end') != original_end:
                    print(f"[强制对齐] 时间戳修正: start {original_start:.3f}->{segment['start']:.3f}, end {original_end:.3f}->{segment['end']:.3f}")

                aligned_segments.append(segment)
                print(f"[强制对齐] 段落 {i+1} 处理完成")

            print(f"[强制对齐] 所有段落处理完成，共 {len(aligned_segments)} 个段落")

            torch.cuda.empty_cache()

            return aligned_segments

        except Exception as e:
            print(f"[强制对齐错误] {str(e)}")
            import traceback
            print(f"[强制对齐错误详情] {traceback.format_exc()}")
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




def _process_cd_segments(cd_result):
    """处理 Whisper-CD 结果的共享函数"""
    result = {
        'segments': [],
        'text': '',
        'language': cd_result['language']
    }

    for segment in cd_result['segments']:
        # 跳过空文本 segment
        text = segment.get('text', '') if isinstance(segment, dict) else getattr(segment, 'text', '')
        if not text or not text.strip():
            continue

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
            'language': cd_result.get('language', ''),
            'temperature': segment.get('temperature', 0.0),
            'avg_logprob': segment.get('avg_logprob', 0.0),
            'compression_ratio': segment.get('compression_ratio', 1.0),
            'no_speech_prob': segment.get('no_speech_prob', 0.0)
        })
        # 根据语言决定文本连接方式：CJK语言无空格，其他语言空格连接
        detected_lang = cd_result.get('language', '')
        if not result['text']:
            result['text'] = text
        else:
            if detected_lang.startswith(('ja', 'zh', 'ko')):
                result['text'] += text
            else:
                result['text'] += ' ' + text

    print(f"[语音识别] CD结果处理完成，提取 {len(result['segments'])} 个段落")
    return result


def _cleanup_memory():
    """清理内存和显存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def recognize_speech_enhanced(audio_path, model_path, detected_language=None, device_choice="auto",
                    progress_callback=None, word_timestamps=True,
                    cd_params: CdParams = None,
                    enable_alignment=True):
    """增强版语音识别

    Args:
        audio_path: 音频文件路径
        model_path: Whisper模型名称或路径
        detected_language: 检测到的语言
        device_choice: 设备选择
        progress_callback: 进度回调
        word_timestamps: 是否启用单词时间戳
        cd_params: CdParams对象，包含所有对比解码参数
        enable_alignment: 是否启用强制对齐

    Returns:
        识别结果字典
    """
    print("[语音识别] 使用增强版语音识别器进行语音识别")

    if cd_params is None:
        cd_params = CdParams()

    device = "cuda" if device_choice != "cpu" else "cpu"

    from config import config
    from utils.whisper_cd_original import WhisperCDOriginal

    print("[Whisper-CD] 启用 Whisper-CD 处理器...")

    whispercd_processor = WhisperCDOriginal(
        model_path=model_path,
        device=device,
        cd_params=cd_params,
        enable_alignment=enable_alignment,
    )

    print("[Whisper-CD] 应用对比解码...")
    cd_result = whispercd_processor.transcribe(
        audio_path,
        detected_language,
        progress_callback=progress_callback
    )

    result = _process_cd_segments(cd_result)

    whispercd_processor.cleanup()
    del whispercd_processor
    _cleanup_memory()

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[内存管理] Whisper-CD 卸载后 GPU 状态: 已分配 {allocated:.2f}GB, 已保留 {reserved:.2f}GB")

    if progress_callback:
        progress_callback(80)

    if enable_alignment and result.get('segments'):
        aligner = None
        try:
            print("[强制对齐] 启用强制对齐...")
            aligner = ForcedAligner(device=device)
            if aligner.load_alignment_model(detected_language or result.get('language', 'ja')):
                result['segments'] = aligner.align(result['segments'], audio_path, return_char_alignments=True)
                print("[强制对齐] 强制对齐完成")
            else:
                print("[强制对齐] 强制对齐模型加载失败，跳过对齐")
        except Exception as e:
            print(f"[强制对齐] 强制对齐失败: {str(e)}")
        finally:
            if aligner is not None:
                aligner.cleanup()
                print("[内存管理] 强制对齐模型已卸载，清理显存")
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    reserved = torch.cuda.memory_reserved() / 1024**3
                    print(f"[内存管理] 强制对齐卸载后 GPU 状态: 已分配 {allocated:.2f}GB, 已保留 {reserved:.2f}GB")

    print("[内存管理] 转录完成，执行最终内存清理...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[内存管理] 最终清理后 GPU 状态: 已分配 {allocated:.2f}GB, 已保留 {reserved:.2f}GB")

    return result


def clear_model_cache():
    """清理模型缓存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[缓存] 模型缓存已清理，GPU: 已分配 {allocated:.2f}GB, 已保留 {reserved:.2f}GB")
