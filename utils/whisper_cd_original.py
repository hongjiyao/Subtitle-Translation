"""
Whisper-CD: 基于论文《Whisper-CD: Accurate Long-Form Speech Recognition using Multi-Negative Contrastive Decoding》的实现
严格按照论文中的方法实现，包括：
1. 30秒分段的长音频处理
2. 启用前文条件预测
3. 贪婪解码
4. 禁用温度回退机制
5. 基于logits的对比解码
6. 三个扰动策略：高斯噪声注入、静音信号、音频时间移位
7. log-sum-exp聚合多个负样本



"""

import os
import math
import zlib
import numpy as np
import torch
import soundfile as sf
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.generation.logits_process import LogitsProcessor
import time
from typing import List, Dict, Any, Optional, Tuple, Callable


os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from utils.speech_recognizer import ForcedAligner
from utils.video_processor import find_ffmpeg
from config import config, CdParams

class WhisperRepetitionSuppressionLogitsProcessor(LogitsProcessor):
    """
    Whisper专用三合一LogitsProcessor

    1. 单token重复：同一非timestamp token在当前segment内出现≥3次 → 设为-inf
    2. n-gram重复：同一3-gram在当前segment内出现≥2次 → 抑制下一个token
    3. 长序列重复检测：滑动窗口比较已生成token序列，检测长句重复并抑制

    关键设计：遇到timestamp token时重置计数器，实现segment边界隔离
    """

    def __init__(self, timestamp_begin=50364, sot_token_id=50257, max_token_repeat=3,
                 ngram_size=3, max_ngram_repeat=2, long_seq_window=20, long_seq_threshold=0.7):
        self.timestamp_begin = timestamp_begin
        self.sot_token_id = sot_token_id
        self.max_token_repeat = max_token_repeat
        self.ngram_size = ngram_size
        self.max_ngram_repeat = max_ngram_repeat
        self.long_seq_window = long_seq_window
        self.long_seq_threshold = long_seq_threshold
        # 当前segment内的text tokens
        self._segment_tokens = []
        # 上一次处理的token位置（用于检测新token）
        self._last_position = 0
        self._sot_found = False  # 是否已找到SOT token

    def __call__(self, input_ids, scores):
        # 获取最新生成的token（每步只生成1个新token）
        if input_ids.shape[1] > self._last_position:
            new_tokens = input_ids[0, self._last_position:].tolist()
            self._last_position = input_ids.shape[1]

            for tid in new_tokens:
                # 在SOT之前的是prompt tokens，不计入重复检测
                if not self._sot_found:
                    # 检测SOT token
                    if tid == self.sot_token_id:
                        self._sot_found = True
                    continue  # 跳过SOT之前的所有token（prompt tokens）

                if tid >= self.timestamp_begin:
                    # 遇到timestamp token，重置segment计数器
                    self._segment_tokens = []
                else:
                    self._segment_tokens.append(tid)

        # 1. 单token重复检测
        token_counts = {}
        for tid in self._segment_tokens:
            token_counts[tid] = token_counts.get(tid, 0) + 1

        for tid, count in token_counts.items():
            if count >= self.max_token_repeat:
                scores[0, tid] = float('-inf')

        # 2. n-gram重复检测
        if len(self._segment_tokens) >= self.ngram_size * self.max_ngram_repeat:
            ngram_counts = {}
            for i in range(len(self._segment_tokens) - self.ngram_size + 1):
                ngram = tuple(self._segment_tokens[i:i + self.ngram_size])
                ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1

            # 当前前缀
            if len(self._segment_tokens) >= self.ngram_size - 1:
                prefix = tuple(self._segment_tokens[-(self.ngram_size - 1):])
                for ngram, count in ngram_counts.items():
                    if count >= self.max_ngram_repeat and ngram[:-1] == prefix:
                        next_token = ngram[-1]
                        if next_token < self.timestamp_begin:
                            scores[0, next_token] = float('-inf')

        # 3. 长序列重复检测
        if len(self._segment_tokens) >= self.long_seq_window * 2:
            recent = self._segment_tokens[-self.long_seq_window:]
            # 与之前的每个窗口比较
            for start in range(0, len(self._segment_tokens) - self.long_seq_window * 2 + 1):
                window = self._segment_tokens[start:start + self.long_seq_window]
                # 计算重合度
                matches = sum(1 for a, b in zip(recent, window) if a == b)
                similarity = matches / self.long_seq_window
                if similarity >= self.long_seq_threshold:
                    # Only suppress the next token (last in recent window)
                    next_token_idx = len(recent) - 1
                    tid = recent[next_token_idx]
                    if tid == window[next_token_idx] and tid < self.timestamp_begin:
                        scores[0, tid] = float('-inf')
                    break  # 只需检测到一次重复

        return scores


class WhisperCDOriginal:
    """严格按照论文实现的Whisper-CD处理器"""

    def __init__(self, model_path: str, device: str = "auto",
                 cd_params: CdParams = None, enable_alignment: bool = True):
        """初始化Whisper-CD处理器

        Args:
            model_path: 模型路径
            device: 设备
            cd_params: 对比解码参数
            enable_alignment: 是否启用强制对齐
        """
        self.model_path = model_path
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        if cd_params is None:
            cd_params = CdParams()
        self.cd_params = cd_params
        self.alpha = cd_params.alpha
        self.temperature = cd_params.temperature
        self.snr_db = cd_params.snr_db
        self.temporal_shift = cd_params.temporal_shift
        self.enable_alignment = enable_alignment
        self.context_max_tokens = cd_params.ctx_tokens

        local_model_path = model_path

        if os.path.isdir(model_path):
            local_model_path = model_path
        elif model_path in ["tiny", "small", "medium", "base", "large", "large-v2", "large-v3", "large-v3-turbo"]:
            model_patterns = {
                "tiny": "tiny",
                "small": "small",
                "medium": "medium",
                "base": "base",
                "large": "large-v3-turbo",
                "large-v2": "large-v2",
                "large-v3": "large-v3-turbo",
                "large-v3-turbo": "large-v3-turbo"
            }

            if model_path in model_patterns:
                potential_path = os.path.join("models", f"openai--whisper-{model_patterns[model_path]}")
                if os.path.isdir(potential_path):
                    local_model_path = potential_path
        elif "large-v3-turbo" in model_path:
            local_v3_turbo_path = os.path.join("models", "openai--whisper-large-v3-turbo")
            if os.path.exists(os.path.join(local_v3_turbo_path, "model.safetensors")):
                local_model_path = local_v3_turbo_path

        if not os.path.isdir(local_model_path):
            raise FileNotFoundError(f"本地模型不存在: {local_model_path}\n请先下载模型到models目录")

        print(f"加载Whisper模型: {local_model_path}")
        self.use_logits = True
        device_map = "cuda" if self.device == "cuda" else None
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.whisper_model = WhisperForConditionalGeneration.from_pretrained(
            local_model_path,
            device_map=device_map,
            dtype=dtype,
            local_files_only=True,
            attn_implementation="sdpa"
        )
        self.whisper_model.to(self.device)
        self.whisper_processor = WhisperProcessor.from_pretrained(local_model_path, local_files_only=True)
        print(f"[模型] Whisper模型加载完成，设备: {self.device}")

    def _load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """加载音频文件或从视频中提取音频

        Args:
            audio_path: 音频路径或视频路径

        Returns:
            音频数据和采样率
        """
        import subprocess
        import tempfile

        ext = os.path.splitext(audio_path)[1].lower()

        if ext in ['.mp4', '.mkv', '.avi', '.mov', '.wmv', '.ts']:
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            temp_audio = temp_file.name
            temp_file.close()

            try:
                subprocess.run(
                    [find_ffmpeg(), "-i", audio_path, "-ac", "1", "-ar", "16000", "-y", temp_audio],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                audio, sr = sf.read(temp_audio)
                os.remove(temp_audio)
                print(f"[音频] 加载完成，时长: {len(audio)/sr:.2f}s，采样率: {sr}")
                return audio, sr
            except Exception as e:
                print(f"[音频加载] FFmpeg音频提取失败: {e}")
                if os.path.exists(temp_audio):
                    os.remove(temp_audio)
                raise e
        else:
            audio, sr = sf.read(audio_path)

            if sr != 16000:
                import torchaudio
                waveform = torch.from_numpy(audio).float()
                if waveform.ndim == 1:
                    waveform = waveform.unsqueeze(0)
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
                waveform = resampler(waveform)
                audio = waveform.squeeze().numpy()
                sr = 16000

            print(f"[音频] 加载完成，时长: {len(audio)/sr:.2f}s，采样率: {sr}")
            return audio, sr

    def _generate_perturbations(self, audio: np.ndarray, sr: int) -> List[np.ndarray]:
        """生成三种扰动音频

        Args:
            audio: 原始音频数据
            sr: 采样率

        Returns:
            扰动音频列表 [高斯噪声, 静音, 时间移位]
        """
        perturbations = []

        # 1. 高斯噪声注入
        noise_std = np.sqrt(np.mean(audio**2) / (10 ** (self.snr_db / 10)))
        noise = np.random.normal(0, noise_std, audio.shape)
        noisy_audio = audio + noise
        perturbations.append(noisy_audio)

        # 2. 静音信号（完全静音）
        silence_audio = np.zeros_like(audio)
        perturbations.append(silence_audio)

        # 3. 音频时间移位
        shift_samples = int(self.temporal_shift * sr)
        if shift_samples < len(audio):
            shifted_audio = np.zeros_like(audio)
            shifted_audio[:-shift_samples] = audio[shift_samples:]
        else:
            shifted_audio = np.zeros_like(audio)
        perturbations.append(shifted_audio)

        return perturbations

    @staticmethod
    def _compute_compression_ratio(text: str) -> float:
        """计算文本的 gzip 压缩比，用于检测重复幻觉"""
        if not text or len(text) < 4:
            return 1.0
        text_bytes = text.encode('utf-8')
        compressed = zlib.compress(text_bytes)
        return max(1.0, len(text_bytes) / len(compressed))

    def _trim_prompt_tokens(self, sequences, prompt_ids):
        """裁剪 sequences 中 prompt_ids 对应的 token，防止解码时包含前文上下文"""
        if prompt_ids is None:
            return sequences
        sot_token_id = self.whisper_processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
        if hasattr(sequences, 'cpu'):
            seq_list = sequences.cpu().tolist()
        elif hasattr(sequences, 'tolist'):
            seq_list = sequences.tolist()
        else:
            seq_list = sequences
        trimmed = []
        for seq in seq_list:
            try:
                sot_pos = seq.index(sot_token_id)
                trimmed.append(seq[sot_pos:])
            except ValueError:
                print(f"[DEBUG] [解码] 序列中未找到 SOT token，跳过裁剪")
                trimmed.append(seq)
        return trimmed

    def _compute_avg_logprob_from_outputs(self, outputs, sequences):
        """从 ModelOutput 的 scores 计算 avg_logprob
        
        当 force_unique_generate_call=True 时，generate 返回 ModelOutput，
        其中 scores 是每步的 logits tuple，sequences 是生成的 token 序列。
        """
        avg_logprob = 0.0
        try:
            scores = outputs.get("scores", None) if isinstance(outputs, dict) else getattr(outputs, "scores", None)
            if scores is None or len(scores) == 0:
                return 0.0

            # sequences: (batch_size, seq_len)
            if hasattr(sequences, 'cpu'):
                token_ids = sequences[0].cpu().tolist()
            elif hasattr(sequences, 'tolist'):
                token_ids = sequences[0].tolist()
            else:
                token_ids = list(sequences[0])

            # 过滤掉 timestamp token 和特殊 token
            timestamp_begin = self.whisper_processor.tokenizer.convert_tokens_to_ids("<|0.00|>")
            if timestamp_begin is None:
                timestamp_begin = 50364
            eos_token_id = self.whisper_processor.tokenizer.eos_token_id

            import torch.nn.functional as F
            log_prob_sum = 0.0
            count = 0
            # scores only contains logits for generated tokens, not decoder input tokens
            # sequences contains [decoder_input_tokens | generated_tokens]
            # Align by taking only the last len(scores) tokens from sequences
            num_scores = len(scores)
            if num_scores > 0 and num_scores <= len(token_ids):
                generated_token_ids = token_ids[-num_scores:]
            else:
                generated_token_ids = token_ids

            for step_idx, token_id in enumerate(generated_token_ids):
                if step_idx >= len(scores):
                    break
                # 跳过特殊 token
                if token_id >= timestamp_begin or token_id == eos_token_id:
                    continue
                score = scores[step_idx]
                if hasattr(score, 'softmax'):
                    log_probs = F.log_softmax(score.float(), dim=-1)
                    if token_id < log_probs.shape[-1]:
                        log_prob_sum += log_probs[0, token_id].item()
                        count += 1
            if count > 0:
                avg_logprob = log_prob_sum / count
        except Exception as e:
            print(f"[DEBUG] [指标提取] avg_logprob 从 ModelOutput 计算失败: {e}")
        return avg_logprob

    def _compute_avg_logprob_for_range(self, outputs, sequences, token_start, token_end):
        """计算指定 token 范围内的平均 logprob"""
        try:
            scores = outputs.get('scores', []) if isinstance(outputs, dict) else getattr(outputs, 'scores', [])
            if not scores:
                return 0.0

            # 获取 token id 列表
            if hasattr(sequences, 'cpu'):
                token_ids = sequences[0].cpu().tolist()
            elif isinstance(sequences, (list, tuple)):
                first_seq = sequences[0]
                if hasattr(first_seq, 'cpu'):
                    token_ids = first_seq.cpu().tolist()
                else:
                    token_ids = list(first_seq)
            else:
                token_ids = list(sequences[0])

            num_scores = len(scores)
            # generated_token_ids 对齐 scores：取最后 num_scores 个 token
            generated_token_ids = token_ids[-num_scores:]

            # 计算在 token_start 到 token_end 范围内的平均 logprob
            log_prob_sum = 0.0
            count = 0
            for step_idx in range(num_scores):
                # 检查该 step 对应的 token 是否在目标范围内
                token_pos = len(token_ids) - num_scores + step_idx
                if token_pos < token_start or token_pos >= token_end:
                    continue

                token_id = generated_token_ids[step_idx]
                if token_id >= scores[step_idx].shape[-1]:
                    continue
                # 跳过 timestamp token
                timestamp_begin = getattr(self, '_timestamp_begin_cache', 50364)
                if token_id >= timestamp_begin:
                    continue

                log_probs = torch.nn.functional.log_softmax(scores[step_idx].float(), dim=-1)
                log_prob_sum += log_probs[0, token_id].item()
                count += 1

            return log_prob_sum / count if count > 0 else 0.0
        except Exception as e:
            print(f"[DEBUG] [指标提取] 按范围计算 avg_logprob 失败: {e}")
            return 0.0

    def _split_long_segment_by_tokens(self, seg, max_duration=4.0, language=None):
        """将时长过长的 segment 按token级别的标点分割

        利用segment的_token_ids字段，在标点token处分割文本，
        按token数量比例分配时间戳（比字符长度比例更精确）。

        Args:
            seg: 单个 segment dict，必须包含 _token_ids 字段
            max_duration: 超过此时长（秒）才分割

        Returns:
            list of dict: 分割后的 segment 列表
        """
        duration = seg['end'] - seg['start']
        if duration <= max_duration or not seg.get('text'):
            return [seg]

        token_ids = seg.get('_token_ids', [])
        if not token_ids:
            # 没有 token_ids 信息，退回不分割
            return [seg]

        tokenizer = self.whisper_processor.tokenizer
        timestamp_begin = getattr(self, '_timestamp_begin_cache', 50364)

        # 构建标点token ID集合（缓存到实例变量避免重复计算）
        if not hasattr(self, '_punct_token_ids_cache'):
            sentence_punct_chars = set('。！？…；!?')  # 句读级（不含空格）
            comma_punct_chars = set('、，,')  # 逗号级
            sentence_punct_ids = set()
            comma_punct_ids = set()
            vocab_size = tokenizer.vocab_size
            for tid in range(vocab_size):
                if tid >= timestamp_begin:
                    continue
                try:
                    decoded = tokenizer.decode([tid])
                    if any(ch in sentence_punct_chars for ch in decoded):
                        sentence_punct_ids.add(tid)
                    if any(ch in comma_punct_chars for ch in decoded):
                        comma_punct_ids.add(tid)
                except Exception:
                    pass
            self._punct_token_ids_cache = (sentence_punct_ids, comma_punct_ids)
            print(f"[TOKEN分割] 标点token缓存构建完成: 句读级{len(sentence_punct_ids)}个, 逗号级{len(comma_punct_ids)}个")

        sentence_punct_ids, comma_punct_ids = self._punct_token_ids_cache

        # 过滤掉 timestamp token，只保留文本 token
        text_token_ids = [tid for tid in token_ids if tid < timestamp_begin]

        if not text_token_ids:
            return [seg]

        # 一级分割：在句读级标点token处分割
        split_positions = []
        for i, tid in enumerate(text_token_ids):
            if tid in sentence_punct_ids:
                split_positions.append(i + 1)  # 标点token归入前一段

        # 二级分割：对一级分割后仍超长的子段，在逗号级标点token处补充分割
        # 先收集所有逗号级分割点
        comma_positions = []
        for i, tid in enumerate(text_token_ids):
            if tid in comma_punct_ids:
                comma_positions.append(i + 1)

        # 如果有句读分割点，检查每个子段是否仍超长，若是则加入逗号分割点
        if split_positions and comma_positions:
            boundaries = [0] + sorted(split_positions) + [len(text_token_ids)]
            for b_idx in range(len(boundaries) - 1):
                sub_start = boundaries[b_idx]
                sub_end = boundaries[b_idx + 1]
                sub_token_count = sub_end - sub_start
                # 估算子段时长比例
                sub_duration_ratio = sub_token_count / len(text_token_ids)
                sub_duration = duration * sub_duration_ratio
                if sub_duration > max_duration:
                    # 在该子段范围内添加逗号分割点
                    for cp in comma_positions:
                        if sub_start < cp < sub_end:
                            split_positions.append(cp)
        elif not split_positions:
            # 无句读分割点时，直接使用逗号分割点
            split_positions = comma_positions

        # 三级分割：无任何标点时，按固定token数分割，寻找助词token等自然断点
        if len(split_positions) == 0:
            # 助词字符集合（可配置）
            particle_chars = set(self.cd_params.particle_chars) if self.cd_params.particle_chars else set()
            is_cjk = language and language.startswith(('ja', 'zh', 'ko'))
            target_token_count = 15  # 约25字符对应的token数
            remaining_tokens = text_token_ids
            offset = 0
            while len(remaining_tokens) > target_token_count:
                best_pos = target_token_count
                best_priority = -1
                # 在目标位置附近寻找自然断点
                search_range = min(6, target_token_count)
                for delta in range(search_range):
                    for pos in [target_token_count + delta, target_token_count - delta]:
                        if 0 < pos < len(remaining_tokens):
                            # 解码该token判断类型
                            try:
                                decoded = tokenizer.decode([remaining_tokens[pos - 1]])
                                priority = -1
                                if any(ch in decoded for ch in '。！？…；!?・'):
                                    priority = 4
                                elif any(ch in decoded for ch in '、，,'):
                                    priority = 3
                                else:
                                    # 纯空格token仅对中日韩语言作为断点
                                    non_space = decoded.replace(' ', '').replace('\u3000', '')
                                    if not non_space and any(ch in decoded for ch in ' \u3000') and is_cjk:
                                        priority = 2
                                    elif any(ch in decoded for ch in particle_chars):
                                        # 排除"よう"中的"よ"
                                        if 'よ' in decoded and pos < len(remaining_tokens):
                                            next_decoded = tokenizer.decode([remaining_tokens[pos]])
                                            if 'う' in next_decoded:
                                                priority = -1
                                            else:
                                                priority = 1
                                        else:
                                            priority = 1
                                if priority > best_priority:
                                    best_priority = priority
                                    best_pos = pos
                            except Exception:
                                pass
                    if best_priority >= 2:
                        break
                split_positions.append(offset + best_pos)
                remaining_tokens = remaining_tokens[best_pos:]
                offset += best_pos

        if not split_positions:
            return [seg]

        # 构建分割后的子segment
        total_tokens = len(text_token_ids)
        result = []
        prev_pos = 0
        for split_pos in sorted(set(split_positions)):
            if split_pos <= prev_pos or split_pos >= total_tokens:
                continue
            part_token_ids = text_token_ids[prev_pos:split_pos]
            part_text = tokenizer.decode(part_token_ids, skip_special_tokens=True).strip()
            if part_text:
                # 按token数量比例分配时间戳
                token_ratio = (split_pos - prev_pos) / total_tokens
                part_duration = duration * token_ratio
                part_start = seg['start'] + duration * (prev_pos / total_tokens)
                part_end = part_start + part_duration
                # 按token位置比例计算 token_start/token_end
                seg_token_start = seg.get('token_start', 0)
                seg_token_end = seg.get('token_end', 0)
                token_range = seg_token_end - seg_token_start
                sub_token_start = seg_token_start + int(prev_pos / total_tokens * token_range)
                sub_token_end = seg_token_start + int(split_pos / total_tokens * token_range)
                result.append({
                    'text': part_text,
                    'start': round(part_start, 3),
                    'end': round(part_end, 3),
                    'words': [],
                    'chars': [],
                    '_token_ids': part_token_ids,
                    'token_start': sub_token_start,
                    'token_end': sub_token_end,
                })
            prev_pos = split_pos

        # 最后一段
        if prev_pos < total_tokens:
            part_token_ids = text_token_ids[prev_pos:]
            part_text = tokenizer.decode(part_token_ids, skip_special_tokens=True).strip()
            if part_text:
                part_start = seg['start'] + duration * (prev_pos / total_tokens)
                # 按token位置比例计算 token_start/token_end
                seg_token_start = seg.get('token_start', 0)
                seg_token_end = seg.get('token_end', 0)
                token_range = seg_token_end - seg_token_start
                sub_token_start = seg_token_start + int(prev_pos / total_tokens * token_range)
                result.append({
                    'text': part_text,
                    'start': round(part_start, 3),
                    'end': round(seg['end'], 3),
                    'words': [],
                    'chars': [],
                    '_token_ids': part_token_ids,
                    'token_start': sub_token_start,
                    'token_end': seg_token_end,
                })

        if len(result) <= 1:
            return [seg]

        # 确保最后一段的end等于原segment的end
        if result:
            result[-1]['end'] = seg['end']

        return result

    def _merge_cross_boundary_segments(self, segments, gap_threshold=2.0, max_duration=4.0, language=None):
        """合并跨30秒chunk边界的断裂句子

        Whisper按30秒固定分段转录，句子如果恰好在边界处会被硬切断。
        例如"風のようなかすかな声が"被切为"風のようなか"和"すかな声が"。
        此方法检测相邻segment的时间间隔和文本连续性，合并断裂的句子。

        Args:
            segments: 解析出的 segment 列表
            gap_threshold: 最大时间间隔（秒），间隔小于此值才考虑合并
            max_duration: 合并后的最大时长（秒），超过则不合并
            language: 语言代码，用于判断是否需要添加空格分隔
        """
        if not segments:
            return segments

        # 句读级标点集合（末尾有这些标点说明句子完整，不需要合并）
        sentence_end_chars = set('。！？…；!?')

        result = [segments[0].copy()]
        for seg in segments[1:]:
            prev = result[-1]
            gap = seg['start'] - prev['end']
            merged_duration = seg['end'] - prev['start']

            # 检查是否需要合并：
            # 1. 时间间隔小于阈值（说明是30秒边界切割导致的断裂）
            # 2. 前一个segment末尾不是句读标点（说明句子未结束）
            # 3. 合并后时长不超过最大时长
            prev_text = prev.get('text', '') if isinstance(prev, dict) else getattr(prev, 'text', '')
            should_merge = (
                gap < gap_threshold
                and prev_text
                and prev_text[-1] not in sentence_end_chars
                and merged_duration <= max_duration
            )

            if should_merge:
                is_cjk = language and language.startswith(('ja', 'zh', 'ko'))
                separator = '' if is_cjk else ' '
                prev['text'] = prev['text'] + separator + seg['text']
                prev['end'] = seg['end']
                if '_token_ids' in prev and '_token_ids' in seg:
                    prev['_token_ids'] = prev['_token_ids'] + seg['_token_ids']
            else:
                result.append(seg.copy())

        return result

    def _merge_short_segments(self, segments, min_duration=2.0, max_duration=15.0, language=None):
        """合并过短的相邻 segment，避免字幕碎片化

        Whisper 日语转录的 timestamp token 是词级别的，解析出的 segment 非常碎
        （如「言葉も」「想いも」「全部」各为一条）。此方法将时长 < min_duration
        的 segment 与下一个 segment 合并，直到达到合理的句子长度。

        Args:
            segments: 解析出的 segment 列表
            min_duration: 最小 segment 时长（秒），低于此值则与下一个合并
            max_duration: 合并后的最大时长（秒），超过则不再合并
            language: 语言代码，用于判断是否需要添加空格分隔
        """
        if not segments:
            return segments

        result = [segments[0].copy()]
        for seg in segments[1:]:
            prev = result[-1]
            prev_duration = prev['end'] - prev['start']
            merged_duration = seg['end'] - prev['start']

            # 如果前一个 segment 太短，且合并后不超过最大时长，则合并
            # 但如果前段以句读标点结尾（完整句子），则不合并
            sentence_end_chars = set('。！？…；!?')
            prev_text = prev.get('text', '')
            prev_ends_with_sentence = prev_text and prev_text[-1] in sentence_end_chars

            if prev_duration < min_duration and merged_duration <= max_duration and not prev_ends_with_sentence:
                # 非CJK语言合并时添加空格分隔
                is_cjk = language and language.startswith(('ja', 'zh', 'ko'))
                separator = '' if is_cjk else ' '
                prev['text'] = prev['text'] + separator + seg['text']
                prev['end'] = seg['end']
                # 拼接 _token_ids（若存在）
                if '_token_ids' in prev and '_token_ids' in seg:
                    prev['_token_ids'] = prev['_token_ids'] + seg['_token_ids']
            else:
                result.append(seg.copy())

        return result

    def _parse_timestamps_from_sequence(self, sequences, original_audio_length, language=None):
        """从 Whisper 输出的 sequences 中解析句子级时间戳

        当 force_unique_generate_call=True 时，generate 返回 ModelOutput，
        只有 sequences 和 scores，没有 segments。此方法从 sequences 中的
        timestamp token 解析出句子级的时间戳和文本。

        Args:
            sequences: 已经过 _trim_prompt_tokens 处理的序列（列表或张量）
            original_audio_length: 原始音频时长（秒）

        Returns:
            list of dict: [{"text": ..., "start": ..., "end": ..., "words": [], "chars": []}]
        """
        time_precision = 0.02
        timestamp_begin = self.whisper_processor.tokenizer.convert_tokens_to_ids("<|0.00|>")
        if timestamp_begin is None:
            timestamp_begin = 50364
        self._timestamp_begin_cache = timestamp_begin

        # Build set of special token IDs to filter from _token_ids
        special_token_ids = set()
        for token_name in ["<|startoftranscript|>", "<|transcribe|>", "<|translate|>", "<|notimestamps|>"]:
            tid = self.whisper_processor.tokenizer.convert_tokens_to_ids(token_name)
            if tid is not None:
                special_token_ids.add(tid)

        # 获取第一个序列的 token id 列表
        if hasattr(sequences, 'cpu'):
            token_ids = sequences[0].cpu().tolist()
        elif hasattr(sequences, 'tolist'):
            token_ids = sequences[0].tolist()
        elif isinstance(sequences, (list, tuple)):
            first_seq = sequences[0]
            if hasattr(first_seq, 'cpu'):
                token_ids = first_seq.cpu().tolist()
            elif hasattr(first_seq, 'tolist'):
                token_ids = first_seq.tolist()
            else:
                token_ids = list(first_seq)
        else:
            token_ids = list(sequences[0])

        # 找到所有 timestamp token 的位置
        # timestamp token: token_id >= timestamp_begin
        timestamp_mask = [t >= timestamp_begin for t in token_ids]

        # 找到连续的 timestamp token 对（两个相邻的 timestamp token 构成一对）
        # 参考 HuggingFace _retrieve_segment: timestamp_segment_indices = where(timestamp_tokens[:-1] & timestamp_tokens[1:])
        consecutive_indices = []
        for i in range(len(timestamp_mask) - 1):
            if timestamp_mask[i] and timestamp_mask[i + 1]:
                consecutive_indices.append(i + 1)  # 记录第二个 timestamp token 的位置

        segments = []

        if len(consecutive_indices) > 0:
            # consecutive_indices mark positions where two adjacent timestamp tokens appear.
            # Each such position is the start of a new segment boundary.
            # Build segment boundaries: [0, ci[0], ci[1], ..., len(token_ids)]
            boundaries = [0] + list(consecutive_indices) + [len(token_ids)]

            for seg_idx in range(len(boundaries) - 1):
                seg_start_pos = boundaries[seg_idx]
                seg_end_pos = boundaries[seg_idx + 1]
                sliced_tokens = token_ids[seg_start_pos:seg_end_pos]

                if not sliced_tokens:
                    continue

                # Find the first timestamp token in this slice (start time)
                start_ts_idx = None
                for j, tid in enumerate(sliced_tokens):
                    if tid >= timestamp_begin:
                        start_ts_idx = j
                        break

                # Find the last timestamp token in this slice (end time)
                end_ts_idx = None
                for j in range(len(sliced_tokens) - 1, -1, -1):
                    if sliced_tokens[j] >= timestamp_begin:
                        end_ts_idx = j
                        break

                if start_ts_idx is None or end_ts_idx is None or start_ts_idx == end_ts_idx:
                    # No valid timestamp pair in this slice, skip
                    continue

                start_time = float(sliced_tokens[start_ts_idx] - timestamp_begin) * time_precision
                end_time = float(sliced_tokens[end_ts_idx] - timestamp_begin) * time_precision

                # Text tokens are between start_ts_idx and end_ts_idx (exclusive)
                text_tokens = sliced_tokens[start_ts_idx + 1:end_ts_idx]

                # Filter out special tokens from _token_ids
                clean_text_tokens = [tid for tid in text_tokens if tid not in special_token_ids]

                # Decode text
                text = self.whisper_processor.tokenizer.decode(text_tokens, skip_special_tokens=True)
                text = text.replace('\ufffd', '').strip()

                if text:
                    segments.append({
                        "text": text,
                        "start": start_time,
                        "end": end_time,
                        "words": [],
                        "chars": [],
                        "token_start": seg_start_pos,
                        "token_end": seg_end_pos,
                        "_token_ids": clean_text_tokens,
                    })
        else:
            # 没有连续的 timestamp token 对
            # 检查是否有单独的 timestamp token
            timestamp_positions = [i for i, is_ts in enumerate(timestamp_mask) if is_ts]

            if len(timestamp_positions) > 0:
                # 有 timestamp token 但不连续，整个序列作为一个 segment
                # 使用第一个 timestamp 作为 start，最后一个 timestamp 作为 end
                start_time = float(token_ids[timestamp_positions[0]] - timestamp_begin) * time_precision
                last_ts_pos = timestamp_positions[-1]
                if token_ids[last_ts_pos] == timestamp_begin:
                    # 最后一个 timestamp 是 <|0.00|>，使用音频全长
                    end_time = float(original_audio_length)
                else:
                    end_time = float(token_ids[last_ts_pos] - timestamp_begin) * time_precision

                # 解码所有非 timestamp 的文本 token
                text_tokens = [t for t in token_ids if t < timestamp_begin]
                clean_text_tokens = [t for t in text_tokens if t not in special_token_ids]
                text = self.whisper_processor.tokenizer.decode(text_tokens, skip_special_tokens=True)
                text = text.replace('\ufffd', '').strip()

                if text:
                    segments.append({
                        "text": text,
                        "start": start_time,
                        "end": end_time,
                        "words": [],
                        "chars": [],
                        "token_start": 0,
                        "token_end": len(token_ids),
                        "_token_ids": clean_text_tokens,
                    })
            else:
                # 完全没有 timestamp token，整个序列作为一个 segment
                text = self.whisper_processor.tokenizer.decode(token_ids, skip_special_tokens=True)
                text = text.replace('\ufffd', '').strip()

                clean_token_ids = [t for t in token_ids if t not in special_token_ids]

                if text:
                    segments.append({
                        "text": text,
                        "start": 0.0,
                        "end": float(original_audio_length),
                        "words": [],
                        "chars": [],
                        "token_start": 0,
                        "token_end": len(token_ids),
                        "_token_ids": clean_token_ids,
                    })

        # 限制 end 时间不超过音频总长
        for seg in segments:
            if seg["end"] > float(original_audio_length):
                seg["end"] = float(original_audio_length)
            if seg["end"] <= seg["start"]:
                seg["end"] = seg["start"] + 0.1

        # 对所有 duration > max_duration 的 segment 按标点分割
        # 不区分 fallback/非fallback，只看时长
        split_segments = []
        for seg in segments:
            split_segments.extend(self._split_long_segment_by_tokens(seg, max_duration=self.cd_params.max_duration, language=language))
        segments = split_segments

        # 合并过短的 segment，避免字幕碎片化
        segments = self._merge_short_segments(segments, min_duration=self.cd_params.min_duration, max_duration=self.cd_params.max_duration, language=language)

        return segments



    def _decode_segment(self, segment_audio, sr, language=None, context="", original_audio_length=0, temperature=0.0, start_time=0.0, end_time=0.0):
        class ContrastiveLogitsProcessor:
            def __init__(self, model, perturbation_encoder_outputs, alpha=1.0, temperature=1.0, device=None):
                self.model = model
                self.alpha = alpha
                self.temperature = temperature
                self.device = device if device else 'cuda'
                self.stacked_encoder_hidden = torch.cat(perturbation_encoder_outputs, dim=0)
                self.K = len(perturbation_encoder_outputs)
                self.past_key_values = None
                self._step = 0

            def __call__(self, input_ids, logits):
                K = self.K
                if K == 0:
                    return logits

                if self.past_key_values is not None:
                    new_token_ids = input_ids[:, -1:]
                else:
                    new_token_ids = input_ids

                expanded_decoder_ids = new_token_ids.expand(K, -1)

                with torch.no_grad():
                    decoder_outputs = self.model.model.decoder(
                        input_ids=expanded_decoder_ids,
                        encoder_hidden_states=self.stacked_encoder_hidden,
                        past_key_values=self.past_key_values,
                        use_cache=True,
                    )
                    self.past_key_values = decoder_outputs.past_key_values
                    perturbation_logits = self.model.proj_out(decoder_outputs.last_hidden_state)
                    perturbation_logits = perturbation_logits[:, -1, :]

                log_avg_exp = self.temperature * (torch.logsumexp(perturbation_logits / self.temperature, dim=0) - math.log(K))

                contrastive_logits = (1 + self.alpha) * logits - self.alpha * log_avg_exp

                self._step += 1
                return contrastive_logits

        perturbations = self._generate_perturbations(segment_audio, sr)
        all_audios = [segment_audio] + perturbations

        all_inputs = self.whisper_processor(all_audios, sampling_rate=16000, return_tensors="pt", padding=True)
        all_input_features = all_inputs.input_features.to(self.device)
        if self.device == "cuda":
            all_input_features = all_input_features.half()
        if all_input_features.shape[-1] < 3000:
            pad_size = 3000 - all_input_features.shape[-1]
            all_input_features = torch.nn.functional.pad(all_input_features, (0, pad_size), mode="constant", value=0)

        with torch.no_grad():
            encoder_outputs = self.whisper_model.model.encoder(all_input_features)

        clean_encoder_output = encoder_outputs.last_hidden_state[:1]
        perturbation_encoder_outputs = [encoder_outputs.last_hidden_state[i:i+1] for i in range(1, len(all_audios))]

        # 静音检测：计算原始音频与静音扰动编码器输出的余弦相似度
        silence_encoder_output = encoder_outputs.last_hidden_state[2:3]  # 静音扰动是第3个（索引2）
        clean_2d = clean_encoder_output.squeeze(0)  # (seq_len, hidden_dim)
        silence_2d = silence_encoder_output.squeeze(0)  # (seq_len, hidden_dim)
        cosine_sim = torch.nn.functional.cosine_similarity(clean_2d, silence_2d, dim=1).mean().item()
        if cosine_sim > 0.9:
            print(f"[静音检测] 片段 {start_time:.2f}s-{end_time:.2f}s 余弦相似度 {cosine_sim:.4f} > 0.9，跳过（疑似静音/音乐）")
            return [], type('Info', (), {"language": language})()
        print(f"[DEBUG] [静音检测] 片段余弦相似度 {cosine_sim:.4f}，继续解码")

        input_features_clean = clean_encoder_output

        attention_mask = torch.ones(input_features_clean.shape[:2], device=self.device, dtype=torch.long)

        # 无上下文时跳过CD（alpha=0），避免CD在首段产生幻觉（如"第1話"重复）
        # CD需要上下文才能正确工作，无上下文时CD反而会放大模型先验导致幻觉
        effective_alpha = self.alpha
        if not context:
            effective_alpha = 0.0
            print(f"[CD调整] 无上下文片段，alpha从{self.alpha}降至0（跳过CD）")

        contrastive_processor = ContrastiveLogitsProcessor(
            self.whisper_model,
            perturbation_encoder_outputs,
            alpha=effective_alpha,
            temperature=self.temperature,
            device=self.device
        )

        prompt_ids = None
        if context:
            try:
                prompt_ids = self.whisper_processor.get_prompt_ids(context, return_tensors="pt")
                if hasattr(prompt_ids, 'input_ids'):
                    prompt_ids = prompt_ids.input_ids
                prompt_ids = prompt_ids.to(self.device)
                # Ensure 2D shape (1, seq_len)
                if isinstance(prompt_ids, torch.Tensor):
                    if prompt_ids.dim() == 1:
                        prompt_ids = prompt_ids.unsqueeze(0)
                prompt_len = prompt_ids.shape[1]
                print(f"[解码] 构建转录上下文，长度: {prompt_len} tokens")
                max_target_positions = getattr(self.whisper_model.config, 'max_target_positions', 448)
                init_tokens = 4
                max_prompt_len = min(self.context_max_tokens, max_target_positions - init_tokens)
                if prompt_ids.shape[1] > max_prompt_len:
                    prompt_ids = prompt_ids[:, -max_prompt_len:]
                    print(f"[DEBUG] [解码] 上下文超限，截断至最后 {max_prompt_len} tokens")
            except Exception as e:
                print(f"[DEBUG] [解码] 无法获取上下文提示 IDs: {e}")

            if prompt_ids is None and context:
                try:
                    context_ids = self.whisper_processor.tokenizer.encode(context)
                    max_target_positions = getattr(self.whisper_model.config, 'max_target_positions', 448)
                    init_tokens = 4
                    max_prompt_len = min(self.context_max_tokens, max_target_positions - init_tokens)
                    if len(context_ids) > max_prompt_len:
                        context_ids = context_ids[-max_prompt_len:]
                    prompt_ids = torch.tensor([context_ids], dtype=torch.long, device=self.device)
                    print(f"[DEBUG] [解码] 使用 tokenizer 编码上下文，长度: {len(context_ids)} tokens")
                except Exception as e:
                    print(f"[DEBUG] [解码] 上下文编码失败: {e}")

        with torch.no_grad():
            from transformers import SuppressTokensLogitsProcessor, SuppressTokensAtBeginLogitsProcessor
            from transformers.modeling_outputs import BaseModelOutput

            gen_config = self.whisper_model.generation_config
            logits_processor_list = [contrastive_processor]

            saved_suppress_tokens = gen_config.suppress_tokens
            saved_begin_suppress_tokens = gen_config.begin_suppress_tokens
            saved_max_length = gen_config.max_length
            saved_return_dict_in_generate = gen_config.return_dict_in_generate
            saved_output_scores = gen_config.output_scores

            try:
                gen_config.suppress_tokens = None
                gen_config.begin_suppress_tokens = None
                gen_config.max_length = 448
                gen_config.return_dict_in_generate = True
                gen_config.output_scores = True

                # 添加三合一重复抑制LogitsProcessor
                # 获取正确的SOT token ID
                sot_id = self.whisper_processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
                if sot_id is None:
                    sot_id = 50257
                repetition_processor = WhisperRepetitionSuppressionLogitsProcessor(
                    timestamp_begin=getattr(self, '_timestamp_begin_cache', 50364),
                    sot_token_id=sot_id,
                    max_token_repeat=3,
                    ngram_size=3,
                    max_ngram_repeat=2,
                    long_seq_window=20,
                    long_seq_threshold=0.7
                )
                logits_processor_list.append(repetition_processor)

                generate_kwargs = {
                    "encoder_outputs": BaseModelOutput(last_hidden_state=input_features_clean),
                    "attention_mask": attention_mask,
                    "language": language,
                    "task": "transcribe",
                    "max_length": 448,
                    "logits_processor": logits_processor_list,
                    "num_beams": 1,
                    "temperature": temperature,
                    "do_sample": False,
                    "return_timestamps": True,
                    "return_segments": True,
                    "force_unique_generate_call": True,
                }
                if prompt_ids is not None:
                    generate_kwargs["prompt_ids"] = prompt_ids.squeeze(0) if prompt_ids.dim() == 2 else prompt_ids
                print("[DEBUG] [解码] 开始生成转录结果...")
                outputs = self.whisper_model.generate(**generate_kwargs)
            finally:
                gen_config.suppress_tokens = saved_suppress_tokens
                gen_config.begin_suppress_tokens = saved_begin_suppress_tokens
                gen_config.max_length = saved_max_length
                gen_config.return_dict_in_generate = saved_return_dict_in_generate
                gen_config.output_scores = saved_output_scores
            print(f"[DEBUG] [解码] 生成完成，outputs类型: {type(outputs)}")

        # CD效果对比：当CD生效时（alpha>0），额外做一次无CD解码对比
        if effective_alpha > 0:
            try:
                cd_transcription = ""
                if isinstance(outputs, dict) and "sequences" in outputs:
                    cd_sequences = self._trim_prompt_tokens(outputs["sequences"], prompt_ids)
                    cd_transcription = self.whisper_processor.batch_decode(cd_sequences, skip_special_tokens=True)[0].strip()

                with torch.no_grad():
                    from transformers.modeling_outputs import BaseModelOutput
                    # 获取正确的SOT token ID（复用之前获取的sot_id）
                    no_cd_repetition_processor = WhisperRepetitionSuppressionLogitsProcessor(
                        timestamp_begin=getattr(self, '_timestamp_begin_cache', 50364),
                        sot_token_id=sot_id,
                        max_token_repeat=3,
                        ngram_size=3,
                        max_ngram_repeat=2,
                        long_seq_window=20,
                        long_seq_threshold=0.7
                    )
                    no_cd_generate_kwargs = {
                        "encoder_outputs": BaseModelOutput(last_hidden_state=input_features_clean),
                        "attention_mask": attention_mask,
                        "language": language,
                        "task": "transcribe",
                        "max_length": 448,
                        "logits_processor": [no_cd_repetition_processor],
                        "num_beams": 1,
                        "temperature": temperature,
                        "do_sample": False,
                        "return_timestamps": True,
                        "force_unique_generate_call": True,
                    }
                    if prompt_ids is not None:
                        no_cd_generate_kwargs["prompt_ids"] = prompt_ids.squeeze(0) if prompt_ids.dim() == 2 else prompt_ids
                    no_cd_outputs = self.whisper_model.generate(**no_cd_generate_kwargs)
                    if isinstance(no_cd_outputs, dict) and "sequences" in no_cd_outputs:
                        no_cd_seq = self._trim_prompt_tokens(no_cd_outputs["sequences"], prompt_ids)
                    else:
                        no_cd_seq = self._trim_prompt_tokens(no_cd_outputs, prompt_ids)
                    no_cd_transcription = self.whisper_processor.batch_decode(no_cd_seq, skip_special_tokens=True)[0].strip()

                if cd_transcription != no_cd_transcription:
                    print(f"[CD效果] 片段 {start_time:.1f}s-{end_time:.1f}s CD修正了转录:")
                    print(f"  [无CD] {no_cd_transcription[:80]}")
                    print(f"  [有CD] {cd_transcription[:80]}")
                else:
                    print(f"[CD效果] 片段 {start_time:.1f}s-{end_time:.1f}s CD无修正（结果一致）")
            except Exception as e:
                print(f"[CD效果] 对比失败: {e}")
        else:
            print(f"[CD效果] 片段 {start_time:.1f}s-{end_time:.1f}s 跳过CD（无上下文）")

        # 提取 sequences 和指标
        if isinstance(outputs, dict) and "sequences" in outputs:
            sequences = outputs["sequences"]
            sequences = self._trim_prompt_tokens(sequences, prompt_ids)
            transcription = self.whisper_processor.batch_decode(sequences, skip_special_tokens=True)[0]

            # force_unique_generate_call=True 时返回 ModelOutput，无 segments
            # 从 sequences 中解析句子级时间戳
            segments = self._parse_timestamps_from_sequence(sequences, original_audio_length, language=language)
            avg_logprob = self._compute_avg_logprob_from_outputs(outputs, sequences)
            print(f"[DEBUG] [指标提取] avg_logprob={avg_logprob:.4f}, scores数量={len(outputs.get('scores', [])) if isinstance(outputs, dict) else len(getattr(outputs, 'scores', []))}, segments数={len(segments)}")

            # 为每个 segment 计算独立指标
            for seg in segments:
                seg_text = seg.get("text", "")
                seg['no_speech_prob'] = 0.0
                seg['compression_ratio'] = self._compute_compression_ratio(seg_text) if seg_text else 1.0

                # 按 segment 的 token 范围计算独立的 avg_logprob
                token_start = seg.get('token_start', 0)
                token_end = seg.get('token_end', 0)
                seg_avg_logprob = self._compute_avg_logprob_for_range(outputs, sequences, token_start, token_end)
                seg['avg_logprob'] = seg_avg_logprob
                if seg_avg_logprob < -0.5 and seg_text:
                    print(f"[警告] 低置信度转录: '{seg_text[:30]}...' avg_logprob={seg_avg_logprob:.4f} (可能存在误识别)")

                seg['temperature'] = temperature

                # 移除临时字段，不写入最终输出
                seg.pop('token_start', None)
                seg.pop('token_end', None)
            if not segments:
                transcription = transcription.replace('\ufffd', '')
                compression_ratio = self._compute_compression_ratio(transcription) if transcription else 1.0
                fallback_seg = {"text": transcription, "start": 0.0, "end": float(original_audio_length), "words": [], "chars": [], 'no_speech_prob': 0.0, 'compression_ratio': compression_ratio, 'avg_logprob': avg_logprob, 'temperature': temperature}
                # 对fallback segment执行分割/合并，避免30秒整段
                fallback_seg['_token_ids'] = []  # fallback无token信息，不分割
                segments = self._split_long_segment_by_tokens(fallback_seg, max_duration=self.cd_params.max_duration, language=language)
                segments = self._merge_short_segments(segments, min_duration=self.cd_params.min_duration, max_duration=self.cd_params.max_duration, language=language)
        else:
            transcription = self.whisper_processor.batch_decode(outputs, skip_special_tokens=True)[0]
            transcription = transcription.replace('\ufffd', '')
            compression_ratio = self._compute_compression_ratio(transcription) if transcription else 1.0
            fallback_seg = {"text": transcription, "start": 0.0, "end": float(original_audio_length), "words": [], "chars": [], 'no_speech_prob': 0.0, 'compression_ratio': compression_ratio, 'avg_logprob': 0.0, 'temperature': temperature}
            # 对fallback segment执行分割/合并，避免30秒整段
            fallback_seg['_token_ids'] = []  # fallback无token信息，不分割
            segments = self._split_long_segment_by_tokens(fallback_seg, max_duration=self.cd_params.max_duration, language=language)
            segments = self._merge_short_segments(segments, min_duration=self.cd_params.min_duration, max_duration=self.cd_params.max_duration, language=language)

        info = type('Info', (), {"language": language})()

        if contrastive_processor.past_key_values is not None:
            del contrastive_processor.past_key_values
            contrastive_processor.past_key_values = None

        return segments, info

    def contrastive_decoding(self, audio_path: str, language: Optional[str] = None,
                           progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """执行对比解码

        Args:
            audio_path: 音频路径
            language: 语言代码
            progress_callback: 进度回调函数

        Returns:
            解码结果
        """
        if progress_callback:
            progress_callback(10, "加载原始音频...")

        original_audio, sr = self._load_audio(audio_path)
        audio_duration = len(original_audio) / sr
        total_start_time = time.time()

        if progress_callback:
            progress_callback(20, "准备模型...")

        if progress_callback:
            progress_callback(30, "加载原始Whisper模型...")

        processed_segments = []

        try:
            if progress_callback:
                progress_callback(40, "开始按30秒分段处理...")

            chunk_duration = 30.0
            audio_duration = len(original_audio) / sr
            segments_to_process = []
            start_time = 0.0
            while start_time < audio_duration:
                end_time = min(start_time + chunk_duration, audio_duration)
                segments_to_process.append((start_time, end_time))
                start_time = end_time
            total_segments = len(segments_to_process)
            print(f"[DEBUG] [分段] 将音频按30秒分为 {total_segments} 个片段")

            print("[DEBUG] 进入对比解码核心处理...")

            if progress_callback:
                progress_callback(45, "开始逐段处理...")

            for i, (start_time, end_time) in enumerate(segments_to_process):
                seg_start_time = time.time()
                if progress_callback:
                    progress_callback(45 + (i / total_segments) * 50, f"处理片段 {i+1}/{total_segments}...")

                start_sample = int(start_time * sr)
                end_sample = int(end_time * sr)
                segment_audio = original_audio[start_sample:end_sample]

                if len(processed_segments) > 0:
                    context_parts = []
                    total_tokens = 0
                    max_target_positions = getattr(self.whisper_model.config, 'max_target_positions', 448)
                    max_tokens = min(self.context_max_tokens, max_target_positions - 4)

                    for seg in reversed(processed_segments):
                        text = seg.get('text', '') if isinstance(seg, dict) else getattr(seg, 'text', '')
                        if not text:
                            continue

                        text_tokens = seg.get('_token_count', None) if isinstance(seg, dict) else getattr(seg, '_token_count', None)
                        if text_tokens is None:
                            cjk_count = sum(1 for c in text if '\u3000' <= c <= '\u9fff' or '\u3040' <= c <= '\u30ff' or '\uac00' <= c <= '\ud7af')
                            text_tokens = cjk_count * 2 + (len(text) - cjk_count)
                        
                        if total_tokens + text_tokens <= max_tokens:
                            context_parts.insert(0, text)
                            total_tokens += text_tokens
                        else:
                            break

                    # 日语/中文上下文直接拼接（无空格），英语用空格连接
                    if language and language.startswith(('ja', 'zh', 'ko')):
                        context = "".join(context_parts)
                    else:
                        context = " ".join(context_parts)
                    if context:
                        print(f"[上下文] 片段 {i+1}: 使用 {len(context_parts)} 个历史片段，上下文长度: {total_tokens} tokens (限制: {max_tokens} tokens, 配置: {self.context_max_tokens})")
                else:
                    context = ""
                    print(f"[上下文] 片段 {i+1}: 无历史上下文")

                segment_duration = end_time - start_time

                # 论文要求：贪婪解码，禁用温度回退（CD本身已抑制幻觉）
                original_segments_list, original_info = self._decode_segment(
                    segment_audio, sr,
                    language=language, context=context,
                    original_audio_length=segment_duration,
                    temperature=0.0,
                    start_time=start_time,
                    end_time=end_time
                )

                detected_language = language
                if original_info and hasattr(original_info, 'language'):
                    detected_language = original_info.language
                    print(f"[DEBUG] 检测到语言: {detected_language}")

                print(f"[DEBUG] 原始片段数: {len(original_segments_list)}")
                if len(original_segments_list) == 0:
                    print(f"[DEBUG] 警告：片段 ({start_time:.2f}s - {end_time:.2f}s) 没有找到 Whisper 片段")
                    print(f"[DEBUG] 音频片段时长: {end_time - start_time:.2f}秒")
                    empty_segment = {
                        'start': start_time,
                        'end': end_time,
                        'text': '',
                        'words': [],
                        'chars': [],
                        '_token_count': 0
                    }
                    processed_segments.append(empty_segment)
                else:
                    for seg_idx, segment in enumerate(original_segments_list):
                        orig_start = segment.get('start', 0) if isinstance(segment, dict) else getattr(segment, 'start', 0)
                        orig_end = segment.get('end', 0) if isinstance(segment, dict) else getattr(segment, 'end', 0)

                        abs_start = start_time + orig_start
                        abs_end = start_time + orig_end

                        abs_start = max(abs_start, start_time)
                        abs_end = min(abs_end, end_time)

                        if abs_end <= abs_start:
                            abs_end = abs_start + 0.1

                        if isinstance(segment, dict):
                            segment['start'] = abs_start
                            segment['end'] = abs_end
                        else:
                            segment.start = abs_start
                            segment.end = abs_end

                        processed_segments.append(segment)
                        if isinstance(segment, dict):
                            segment['_token_count'] = len(self.whisper_processor.tokenizer.encode(segment.get('text', '')))
                        print(f"[DEBUG] [对比解码] 片段 {i+1}-{seg_idx+1}: Whisper时间戳 {orig_start:.3f}s - {orig_end:.3f}s → 绝对时间 {abs_start:.3f}s - {abs_end:.3f}s (分段: {start_time:.3f}s - {end_time:.3f}s)")

                seg_elapsed = time.time() - seg_start_time
                print(f"[DEBUG] [耗时] 片段 {i+1}/{total_segments} 解码耗时: {seg_elapsed:.2f}s")

            total_elapsed = time.time() - total_start_time
            avg_per_segment = total_elapsed / total_segments if total_segments > 0 else 0
            print(f"[耗时] 总转录耗时: {total_elapsed:.2f}s, 平均每片段: {avg_per_segment:.2f}s, 共 {total_segments} 个片段")

            if progress_callback:
                progress_callback(100, "处理完成")

            language_probability = 1.0

            for seg in processed_segments:
                seg.pop('_token_count', None)

            # 全局后处理：过滤空文本
            final_segments = []
            for seg in processed_segments:
                text = seg.get('text', '') if isinstance(seg, dict) else getattr(seg, 'text', '')
                if text and text.strip():
                    final_segments.append(seg)
                # 空文本segment直接过滤，不传递给下游
            # 跨chunk边界合并断裂句子（在合并短segment之前）
            final_segments = self._merge_cross_boundary_segments(final_segments, gap_threshold=self.cd_params.gap_threshold, max_duration=self.cd_params.max_duration, language=detected_language)
            final_segments = self._merge_short_segments(final_segments, min_duration=self.cd_params.min_duration, max_duration=self.cd_params.max_duration, language=detected_language)

            # 合并操作完成后再移除 _token_ids
            for seg in final_segments:
                seg.pop('_token_ids', None)

            processed_segments = final_segments
            print(f"[全局后处理] 合并+过滤后: {len(processed_segments)} 条字幕")

            result = {
                "segments": processed_segments,
                "language": detected_language,
                "language_probability": language_probability,
                "filtered_segment_count": len(processed_segments)
            }

            if self.enable_alignment:
                print("[DEBUG] [强制对齐] 对齐将由外层 recognize_speech_enhanced 函数执行")

            print(f"[DEBUG] 返回前 segments 数量: {len(result['segments'])}")
            if result['segments']:
                first_seg = result['segments'][0]
                print(f"[DEBUG] 第一个片段: start={first_seg.get('start', 0):.2f}, end={first_seg.get('end', 0):.2f}, text={first_seg.get('text', '')[:30]}")
            return result

        finally:
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def transcribe(self, audio_path: str, language: Optional[str] = None,
                  progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """转录音频

        Args:
            audio_path: 音频路径
            language: 语言代码
            progress_callback: 进度回调函数

        Returns:
            转录结果
        """
        return self.contrastive_decoding(audio_path, language, progress_callback)

    def cleanup(self):
        if hasattr(self, 'whisper_model') and self.whisper_model is not None:
            del self.whisper_model
            self.whisper_model = None
        if hasattr(self, 'whisper_processor') and self.whisper_processor is not None:
            del self.whisper_processor
            self.whisper_processor = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        print("[内存管理] Whisper-CD 模型已卸载，清理显存")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Whisper-CD 论文实现")
    parser.add_argument("audio_path", help="音频文件路径")
    parser.add_argument("--model", default="medium", help="模型名称")
    parser.add_argument("--language", default=None, help="语言代码")
    parser.add_argument("--alpha", type=float, default=0.5, help="对比强度")
    parser.add_argument("--temperature", type=float, default=1.0, help="log-sum-exp温度")
    parser.add_argument("--snr_db", type=float, default=10.0, help="高斯噪声SNR")
    parser.add_argument("--temporal_shift", type=float, default=7.0, help="时间移位秒数")

    args = parser.parse_args()

    processor = WhisperCDOriginal(
        args.model,
        cd_params=CdParams(
            alpha=args.alpha,
            temperature=args.temperature,
            snr_db=args.snr_db,
            temporal_shift=args.temporal_shift
        )
    )

    def progress_callback(progress, message):
        print(f"进度: {progress}% - {message}")

    print(f"正在处理音频: {args.audio_path}")
    result = processor.transcribe(args.audio_path, args.language, progress_callback)

    print(f"检测到语言: {result['language']} (概率: {result['language_probability']:.2f})")
    print("识别结果:")
    for i, segment in enumerate(result['segments']):
        print(f"[{i+1}] {segment['start']:.2f}s - {segment['end']:.2f}s: {segment['text']}")
