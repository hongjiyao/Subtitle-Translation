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
import numpy as np
import torch
import soundfile as sf
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import time
from typing import List, Dict, Any, Optional, Tuple

def clean_token_for_print(token_str):
    """清理token中的无效Unicode字符"""
    if isinstance(token_str, list):
        return [clean_token_for_print(t) for t in token_str]
    if isinstance(token_str, str):
        try:
            return token_str.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
        except:
            return token_str
    return token_str

from utils.speech_recognizer import ForcedAligner
from config import config, CdParams

class WhisperCDOriginal:
    """严格按照论文实现的Whisper-CD处理器"""

    def __init__(self, model_path: str, device: str = "auto", compute_type: str = "float16",
                 cd_params: CdParams = None,
                 batch_size: int = 1, enable_alignment: bool = True,
                 max_duration: float = None, speech_segments: list = None):
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        """初始化Whisper-CD处理器

        Args:
            model_path: 模型路径
            device: 设备
            compute_type: 计算类型
            alpha: 对比强度
            temperature: log-sum-exp温度
            snr_db: 高斯噪声SNR
            temporal_shift: 时间移位秒数
            batch_size: 批量处理大小
            enable_alignment: 是否启用强制对齐
            speech_segments: ten-vad检测到的语音片段列表 [(start, end), ...]
        """
        self.model_path = model_path
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.compute_type = compute_type
        if cd_params is None:
            cd_params = CdParams()
        self.alpha = cd_params.alpha
        self.temperature = cd_params.temperature
        self.snr_db = cd_params.snr_db
        self.temporal_shift = cd_params.temporal_shift
        self.batch_size = batch_size
        self.enable_alignment = enable_alignment
        self.max_duration = max_duration
        self.speech_segments = speech_segments or []
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
            local_files_only=True
        )
        self.whisper_model.to(self.device)
        self.whisper_processor = WhisperProcessor.from_pretrained(local_model_path, local_files_only=True)

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
            temp_audio = tempfile.mktemp(suffix=".wav")

            try:
                subprocess.run(
                    ["ffmpeg", "-i", audio_path, "-ac", "1", "-ar", "16000", "-y", temp_audio],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                audio, sr = sf.read(temp_audio)
                os.remove(temp_audio)
                return audio, sr
            except Exception as e:
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

    def _batch_process(self, audios: List[np.ndarray], sr: int, language: Optional[str] = None, context: str = "") -> List[Any]:
        """批量处理音频（内存中直接处理）

        Args:
            audios: 音频数据列表（内存中的numpy数组）
            sr: 采样率
            language: 语言代码
            context: 上下文文本

        Returns:
            处理结果列表，每个元素是一个元组：(segments, info, audio_path)
        """
        import torch
        results = []

        try:
            original_audio = audios[0]

            print(f"[批量处理] 处理音频长度: {len(original_audio)} 采样点")

            # 预处理原始音频
            inputs = self.whisper_processor(original_audio, sampling_rate=16000, return_tensors="pt")
            input_features = inputs.input_features.to(self.device)
            if self.device == "cuda":
                input_features = input_features.half()
            attention_mask = inputs.get("attention_mask", torch.ones(input_features.shape[:2], device=self.device, dtype=torch.long))
            if attention_mask.device != self.device:
                attention_mask = attention_mask.to(self.device)
            print(f"[批量处理] 原始音频input_features形状: {input_features.shape}")

            # 预处理扰动音频
            perturbation_input_features = []
            for i in range(1, len(audios)):
                perturb_audio = audios[i]
                perturb_inputs = self.whisper_processor(perturb_audio, sampling_rate=16000, return_tensors="pt")
                perturb_feat = perturb_inputs.input_features.to(self.device)
                if self.device == "cuda":
                    perturb_feat = perturb_feat.half()
                perturbation_input_features.append(perturb_feat)
                print(f"[批量处理] 扰动音频{i} input_features形状: {perturb_feat.shape}")

            # 批量编码器前向传播：将clean + K个扰动input_features拼接，一次编码
            all_input_features = torch.cat([input_features] + perturbation_input_features, dim=0)
            print(f"[批量处理] 批量编码器输入形状: {all_input_features.shape}")
            with torch.no_grad():
                encoder_outputs = self.whisper_model.model.encoder(all_input_features)
            clean_encoder_output = encoder_outputs.last_hidden_state[:1]
            perturbation_encoder_outputs = [encoder_outputs.last_hidden_state[i:i+1] for i in range(1, 1 + len(perturbation_input_features))]
            print(f"[批量处理] 编码器输出: clean={clean_encoder_output.shape}, {len(perturbation_encoder_outputs)}个扰动输出")

            # 对比解码的LogitsProcessor
            class ContrastiveLogitsProcessor:
                def __init__(self, model, perturbation_encoder_outputs, alpha=1.0, temperature=1.0, device=None):
                    self.model = model
                    self.perturbation_encoder_outputs = perturbation_encoder_outputs
                    self.alpha = alpha
                    self.temperature = temperature
                    self.device = device if device else 'cuda'

                def __call__(self, input_ids, logits):
                    K = len(self.perturbation_encoder_outputs)
                    if K == 0:
                        return logits

                    # 将K个扰动编码器输出沿batch维度拼接
                    stacked_encoder_hidden = torch.cat(self.perturbation_encoder_outputs, dim=0)

                    # 扩展decoder_input_ids以匹配K条路径（论文：reusing the same prefix tokens y<t）
                    expanded_decoder_ids = input_ids.expand(K, -1)

                    # 单次批量解码器前向传播
                    with torch.no_grad():
                        decoder_outputs = self.model.model.decoder(
                            input_ids=expanded_decoder_ids,
                            encoder_hidden_states=stacked_encoder_hidden,
                        )
                        perturbation_logits = self.model.proj_out(decoder_outputs.last_hidden_state)
                        perturbation_logits = perturbation_logits[:, -1, :]

                    # 计算log-mean-exp聚合（数值稳定实现）
                    max_perturb_logits = torch.max(perturbation_logits, dim=0)[0]
                    scaled_diff = (perturbation_logits - max_perturb_logits.unsqueeze(0)) / self.temperature
                    exp_logits = torch.exp(scaled_diff)
                    avg_exp = torch.mean(exp_logits, dim=0)
                    log_avg_exp = self.temperature * torch.log(avg_exp) + max_perturb_logits

                    # 对比解码公式：contrastive_logits = (1 + alpha) * logits - alpha * log_avg_exp
                    contrastive_logits = (1 + self.alpha) * logits - self.alpha * log_avg_exp

                    return contrastive_logits

            # 创建对比解码的LogitsProcessor
            contrastive_processor = ContrastiveLogitsProcessor(
                self.whisper_model,
                perturbation_encoder_outputs,
                alpha=self.alpha,
                temperature=self.temperature,
                device=self.device
            )

            # 构建 prompt_ids（转录上下文）- 使用Whisper标准机制
            prompt_ids = None
            if context:
                try:
                    prompt_ids = self.whisper_processor.get_prompt_ids(context, return_tensors="pt")
                    if hasattr(prompt_ids, 'input_ids'):
                        prompt_ids = prompt_ids.input_ids
                    prompt_ids = prompt_ids.to(self.device)
                    prompt_len = prompt_ids.shape[0] if hasattr(prompt_ids, 'shape') else len(prompt_ids)
                    print(f"[批量处理] 构建转录上下文，长度: {prompt_len} tokens")
                    max_target_positions = getattr(self.whisper_model.config, 'max_target_positions', 448)
                    init_tokens = 4
                    max_prompt_len = min(self.context_max_tokens, max_target_positions - init_tokens)
                    if prompt_len > max_prompt_len:
                        prompt_ids = prompt_ids[-max_prompt_len:]
                        print(f"[批量处理] 上下文超限，截断至最后 {max_prompt_len} tokens (总预算={max_target_positions}, init={init_tokens}, 上下文限制={self.context_max_tokens})")
                except Exception as e:
                    print(f"[批量处理] 无法获取上下文提示 IDs: {e}")

                if prompt_ids is None and context:
                    try:
                        context_ids = self.whisper_processor.tokenizer.encode(context)
                        max_target_positions = getattr(self.whisper_model.config, 'max_target_positions', 448)
                        init_tokens = 4
                        max_prompt_len = min(self.context_max_tokens, max_target_positions - init_tokens)
                        if len(context_ids) > max_prompt_len:
                            context_ids = context_ids[-max_prompt_len:]
                        prompt_ids = torch.tensor([context_ids], device=self.device)
                        print(f"[批量处理] 使用 tokenizer 编码上下文，长度: {len(context_ids)} tokens")
                    except Exception as e:
                        print(f"[批量处理] 上下文编码失败: {e}")

            # 生成转录结果 - 严格按照论文设置：贪婪解码，禁用温度回退
            with torch.no_grad():
                from transformers import SuppressTokensLogitsProcessor, SuppressTokensAtBeginLogitsProcessor

                gen_config = self.whisper_model.generation_config
                logits_processor_list = [contrastive_processor]

                if gen_config.suppress_tokens is not None:
                    logits_processor_list.append(
                        SuppressTokensLogitsProcessor(gen_config.suppress_tokens, device=self.device)
                    )

                if gen_config.begin_suppress_tokens is not None:
                    begin_index = 3 if hasattr(gen_config, 'forced_bos_token_id') and gen_config.forced_bos_token_id is not None else 2
                    logits_processor_list.append(
                        SuppressTokensAtBeginLogitsProcessor(gen_config.begin_suppress_tokens, begin_index=begin_index, device=self.device)
                    )

                saved_suppress_tokens = gen_config.suppress_tokens
                saved_begin_suppress_tokens = gen_config.begin_suppress_tokens
                saved_max_length = gen_config.max_length
                saved_return_dict_in_generate = gen_config.return_dict_in_generate

                gen_config.suppress_tokens = None
                gen_config.begin_suppress_tokens = None
                gen_config.max_length = None
                gen_config.return_dict_in_generate = True

                generate_kwargs = {
                    "input_features": input_features,
                    "attention_mask": attention_mask,
                    "language": language,
                    "task": "transcribe",
                    "max_length": None,
                    "logits_processor": logits_processor_list,
                    "num_beams": 1,
                    "temperature": 0.0,
                    "do_sample": False,
                    "return_timestamps": True,
                    "return_segments": True,
                }
                if prompt_ids is not None:
                    generate_kwargs["prompt_ids"] = prompt_ids
                print("[批量处理] 开始生成转录结果...")
                import logging
                whisper_logger = logging.getLogger("transformers.models.whisper.generation_whisper")
                saved_logger_level = whisper_logger.level
                whisper_logger.setLevel(logging.ERROR)
                try:
                    outputs = self.whisper_model.generate(**generate_kwargs)
                finally:
                    whisper_logger.setLevel(saved_logger_level)
                    gen_config.suppress_tokens = saved_suppress_tokens
                    gen_config.begin_suppress_tokens = saved_begin_suppress_tokens
                    gen_config.max_length = saved_max_length
                    gen_config.return_dict_in_generate = saved_return_dict_in_generate
                print(f"[批量处理] 生成完成，outputs类型: {type(outputs)}")

            # 解码并提取时间戳
            if isinstance(outputs, dict):
                transcription = self.whisper_processor.batch_decode(outputs["sequences"], skip_special_tokens=True)[0]
                whisper_segments = outputs.get("segments", [])

                if whisper_segments and len(whisper_segments) > 0:
                    seg_data = whisper_segments[0][0] if whisper_segments[0] else {}
                    ws_start = seg_data.get("start", 0)
                    ws_end = seg_data.get("end", len(original_audio)/sr)
                    if hasattr(ws_start, 'item'):
                        ws_start = ws_start.item()
                    if hasattr(ws_end, 'item'):
                        ws_end = ws_end.item()
                    print(f"[批量处理] Whisper时间戳: {ws_start:.3f}s - {ws_end:.3f}s")
                else:
                    ws_start, ws_end = 0, len(original_audio)/sr
                    print(f"[批量处理] 无Whisper时间戳，使用默认值: {ws_start:.3f}s - {ws_end:.3f}s")
            else:
                transcription = self.whisper_processor.batch_decode(outputs, skip_special_tokens=True)[0]
                ws_start, ws_end = 0, len(original_audio)/sr
                print(f"[批量处理] outputs是Tensor，无Whisper时间戳")

            transcription = transcription.replace('\ufffd', '')

            print(f"[批量处理] 转录结果: {transcription[:100]}...")

            segments = [{"text": transcription, "start": ws_start, "end": ws_end}]
            info = type('Info', (), {"language": language})()

            results.append((segments, info, None))

            for _ in range(len(audios) - 1):
                results.append(([], None, None))
        except Exception as e:
            print(f"处理音频时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            for _ in audios:
                results.append(([], None, None))

        import gc
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()

        return results

    def contrastive_decoding(self, audio_path: str, language: Optional[str] = None,
                           progress_callback: Optional[callable] = None) -> Dict[str, Any]:
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

        if self.max_duration and self.max_duration > 0:
            max_samples = int(self.max_duration * sr)
            if len(original_audio) > max_samples:
                original_audio = original_audio[:max_samples]
                audio_duration = len(original_audio) / sr
                print(f"[音频信息] 限制处理时长为: {self.max_duration:.2f} 秒")
                print(f"[音频信息] 实际处理音频长度: {audio_duration:.2f} 秒")

        if progress_callback:
            progress_callback(20, "准备模型...")

        if progress_callback:
            progress_callback(30, "加载原始Whisper模型...")

        processed_segments = []
        original_segment_count = 0
        temp_files = []

        try:
            if progress_callback:
                progress_callback(40, "开始按VAD片段处理...")

            total_vad_segments = len(self.speech_segments)
            if total_vad_segments == 0:
                print("[VAD] 未检测到语音活动，使用整个音频按30秒分块处理")
                audio_duration = len(original_audio) / sr
                chunk_duration = 30.0
                self.speech_segments = []
                for start_time in range(0, int(audio_duration), int(chunk_duration)):
                    end_time = min(start_time + chunk_duration, audio_duration)
                    self.speech_segments.append((start_time, end_time))
                print(f"[VAD] 将音频分为 {len(self.speech_segments)} 个片段")
                total_vad_segments = len(self.speech_segments)

            print("[DEBUG] 进入对比解码核心处理...")

            for i, (start_time, end_time) in enumerate(self.speech_segments):
                if progress_callback:
                    progress_callback(40 + (i / total_vad_segments) * 50, f"处理VAD片段 {i+1}/{total_vad_segments}...")

                start_sample = int(start_time * sr)
                end_sample = int(end_time * sr)
                segment_audio = original_audio[start_sample:end_sample]

                perturbations = self._generate_perturbations(segment_audio, sr)

                if len(processed_segments) > 0:
                    context_parts = []
                    total_tokens = 0
                    max_target_positions = getattr(self.whisper_model.config, 'max_target_positions', 448)
                    max_tokens = min(self.context_max_tokens, max_target_positions - 4)

                    for seg in reversed(processed_segments):
                        text = seg.get('text', '') if isinstance(seg, dict) else getattr(seg, 'text', '')
                        if not text:
                            continue

                        try:
                            text_tokens = len(self.whisper_processor.tokenizer.encode(text))
                        except Exception:
                            text_tokens = len(text)
                        
                        if total_tokens + text_tokens <= max_tokens:
                            context_parts.insert(0, text)
                            total_tokens += text_tokens
                        else:
                            break

                    context = " ".join(context_parts)
                    if context:
                        print(f"[上下文] 片段 {i+1}: 使用 {len(context_parts)} 个历史片段，上下文长度: {total_tokens} tokens (限制: {max_tokens} tokens, 配置: {self.context_max_tokens})")
                else:
                    context = ""
                    print(f"[上下文] 片段 {i+1}: 无历史上下文")

                all_audios = [segment_audio] + perturbations

                results = self._batch_process(all_audios, sr, language, context)

                original_segments, original_info, original_audio_path = results[0]
                original_segments_list = list(original_segments)

                detected_language = language
                if original_info and hasattr(original_info, 'language'):
                    detected_language = original_info.language
                    print(f"[DEBUG] 检测到语言: {detected_language}")

                print(f"[DEBUG] 原始片段数: {len(original_segments_list)}")
                if len(original_segments_list) == 0:
                    print(f"[DEBUG] 警告：VAD片段 ({start_time:.2f}s - {end_time:.2f}s) 没有找到 Whisper 片段")
                    print(f"[DEBUG] 音频片段时长: {end_time - start_time:.2f}秒")
                    empty_segment = {
                        'start': start_time,
                        'end': end_time,
                        'text': '',
                        'words': [],
                        'chars': []
                    }
                    processed_segments.append(empty_segment)
                else:
                    perturbation_results = results[1:]

                    for seg_idx, segment in enumerate(original_segments_list):
                        orig_start = segment.get('start', 0) if isinstance(segment, dict) else getattr(segment, 'start', 0)
                        orig_end = segment.get('end', 0) if isinstance(segment, dict) else getattr(segment, 'end', 0)

                        abs_start = start_time + orig_start
                        abs_end = start_time + orig_end

                        if isinstance(segment, dict):
                            segment['start'] = abs_start
                            segment['end'] = abs_end
                        else:
                            segment.start = abs_start
                            segment.end = abs_end

                        processed_segments.append(segment)
                        print(f"[对比解码] 片段 {i+1}-{seg_idx+1}: Whisper时间戳 {orig_start:.3f}s - {orig_end:.3f}s → 绝对时间 {abs_start:.3f}s - {abs_end:.3f}s")

                for result in results:
                    _, _, temp_path = result
                    if temp_path and os.path.exists(temp_path):
                        try:
                            os.remove(temp_path)
                        except:
                            pass

            if progress_callback:
                progress_callback(100, "处理完成")

            detected_language = language
            language_probability = 1.0
            if 'original_info' in locals() and original_info is not None:
                if hasattr(original_info, 'language'):
                    detected_language = original_info.language
                if hasattr(original_info, 'language_probability'):
                    language_probability = original_info.language_probability

            result = {
                "segments": processed_segments,
                "language": detected_language,
                "language_probability": language_probability,
                "original_segment_count": original_segment_count,
                "filtered_segment_count": len(processed_segments)
            }

            if self.enable_alignment:
                print("[强制对齐] 对齐将由外层 recognize_speech_enhanced 函数执行")

            print(f"[DEBUG] 返回前 segments 数量: {len(result['segments'])}")
            if result['segments']:
                first_seg = result['segments'][0]
                print(f"[DEBUG] 第一个片段: start={first_seg.get('start', 0):.2f}, end={first_seg.get('end', 0):.2f}, text={first_seg.get('text', '')[:30]}")
            return result

        finally:
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except:
                        pass

    def transcribe(self, audio_path: str, language: Optional[str] = None,
                  progress_callback: Optional[callable] = None) -> Dict[str, Any]:
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
            torch.cuda.empty_cache()
        print("[内存管理] Whisper-CD 模型已卸载，清理显存")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Whisper-CD 论文实现")
    parser.add_argument("audio_path", help="音频文件路径")
    parser.add_argument("--model", default="medium", help="模型名称")
    parser.add_argument("--language", default=None, help="语言代码")
    parser.add_argument("--alpha", type=float, default=1.0, help="对比强度")
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
        print(f"[{i+1}] {segment.start:.2f}s - {segment.end:.2f}s: {segment.text}")
