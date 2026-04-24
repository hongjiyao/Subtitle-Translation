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

# 导入增强版语音识别器
from utils.speech_recognizer import EnhancedSpeechRecognizer

class WhisperCDOriginal:
    """严格按照论文实现的Whisper-CD处理器"""

    def __init__(self, model_path: str, device: str = "auto", compute_type: str = "float16",
                 alpha: float = 1.0, temperature: float = 1.0, snr_db: float = 10.0,
                 temporal_shift: float = 7.0, score_threshold: float = 0.1,
                 batch_size: int = 1, enable_alignment: bool = True,
                 max_duration: float = None, speech_segments: list = None,
                 context_segments: int = 10):
        import os
        import torch
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
            score_threshold: 置信度分数阈值
            batch_size: 批量处理大小
            enable_alignment: 是否启用强制对齐
            speech_segments: ten-vad检测到的语音片段列表 [(start, end), ...]
        """
        self.model_path = model_path
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.compute_type = compute_type
        self.alpha = alpha
        self.temperature = temperature
        self.snr_db = snr_db
        self.temporal_shift = temporal_shift
        self.score_threshold = score_threshold
        self.batch_size = batch_size
        self.enable_alignment = enable_alignment
        self.max_duration = max_duration
        self.speech_segments = speech_segments or []
        self.context_segments = context_segments  # 保留多少个片段作为上下文
        
        # 检查本地模型路径
        import os
        
        # 尝试找到本地模型
        local_model_path = model_path
        
        # 检查是否是完整的模型路径
        if os.path.isdir(model_path):
            # 如果是完整路径，直接使用
            local_model_path = model_path
        elif model_path in ["tiny", "small", "medium", "base", "large", "large-v2", "large-v3", "large-v3-turbo"]:
            # 如果模型路径是模型名称，尝试在本地查找
            # 检查models目录
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
            # 对于large-v3-turbo模型，只使用本地模型
            local_v3_turbo_path = os.path.join("models", "openai--whisper-large-v3-turbo")
            if os.path.exists(os.path.join(local_v3_turbo_path, "model.safetensors")):
                local_model_path = local_v3_turbo_path
        
        # 初始化增强版语音识别器（用于强制对齐）
        self.enhanced_recognizer = EnhancedSpeechRecognizer(
            model_path=local_model_path,
            device=device,
            compute_type=compute_type,
            enable_alignment=enable_alignment
        )
        
        # 检查是否是原始Whisper模型（包含model.safetensors）
        if os.path.exists(os.path.join(local_model_path, "model.safetensors")):
            # 加载原始Whisper模型
            print(f"加载原始Whisper模型: {local_model_path}")
            self.use_logits = True
            # 加载模型，根据设备选择合适的配置
            import torch
            device_map = "cuda" if self.device == "cuda" else None
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            
            self.whisper_model = WhisperForConditionalGeneration.from_pretrained(
                local_model_path,
                device_map=device_map,
                dtype=dtype,
                local_files_only=True  # 只使用本地文件
            )
            # 确保模型在正确的设备上
            self.whisper_model.to(self.device)
            self.whisper_processor = WhisperProcessor.from_pretrained(local_model_path, local_files_only=True)
        elif model_path == "large-v3-turbo" or "large-v3-turbo" in model_path:
            # 对于large-v3-turbo模型，只使用本地模型
            local_v3_turbo_path = os.path.join("models", "openai--whisper-large-v3-turbo")
            if os.path.exists(os.path.join(local_v3_turbo_path, "model.safetensors")):
                # 加载本地模型
                print(f"加载本地large-v3-turbo模型: {local_v3_turbo_path}")
                self.use_logits = True
                import torch
                self.whisper_model = WhisperForConditionalGeneration.from_pretrained(
                    local_v3_turbo_path,
                    device_map="cuda",
                    dtype=torch.float16,
                    local_files_only=True
                )
                self.whisper_model.to(self.device)
                self.whisper_processor = WhisperProcessor.from_pretrained(local_v3_turbo_path, local_files_only=True)
            else:
                # 本地没有模型，报错
                raise FileNotFoundError(f"本地模型不存在: {local_v3_turbo_path}\n请先下载模型到models目录")
        else:
            # 对于其他模型，也使用transformers库加载原始Whisper模型
            print(f"加载原始Whisper模型: {model_path}")
            self.use_logits = True
            # 加载模型到GPU，使用fp16精度减少内存使用
            import torch
            self.whisper_model = WhisperForConditionalGeneration.from_pretrained(
                model_path,
                device_map="cuda",  # 明确指定使用GPU
                dtype=torch.float16,
                local_files_only=True  # 只使用本地文件
            )
            # 确保模型在GPU上
            self.whisper_model.to(self.device)
            self.whisper_processor = WhisperProcessor.from_pretrained(model_path, local_files_only=True)

    def _load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """加载音频文件或从视频中提取音频
        
        Args:
            audio_path: 音频路径或视频路径
            
        Returns:
            音频数据和采样率
        """
        import subprocess
        import tempfile
        import os
        
        # 检查文件扩展名
        ext = os.path.splitext(audio_path)[1].lower()
        
        # 如果是视频文件，提取音频
        if ext in ['.mp4', '.mkv', '.avi', '.mov', '.wmv', '.ts']:
            # 创建临时音频文件
            temp_audio = tempfile.mktemp(suffix=".wav")
            
            # 使用ffmpeg提取音频
            try:
                subprocess.run(
                    ["ffmpeg", "-i", audio_path, "-ac", "1", "-ar", "16000", "-y", temp_audio],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                # 读取提取的音频
                audio, sr = sf.read(temp_audio)
                # 清理临时文件
                os.remove(temp_audio)
                return audio, sr
            except Exception as e:
                # 清理临时文件
                if os.path.exists(temp_audio):
                    os.remove(temp_audio)
                raise e
        else:
            # 直接读取音频文件
            audio, sr = sf.read(audio_path)

            # 如果采样率不是16000，重采样到16000
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
        noisy_audio = np.clip(noisy_audio, -1, 1)
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
            # 处理原始音频（第一个音频）
            original_audio = audios[0]
            
            # 直接使用内存中的音频数据，不需要临时文件
            print(f"[批量处理] 处理音频长度: {len(original_audio)} 采样点")
            
            # 预处理原始音频
            inputs = self.whisper_processor(original_audio, sampling_rate=16000, return_tensors="pt")
            # 将输入张量转换为fp16精度，与模型精度一致
            input_features = inputs.input_features.to(self.device).half()
            print(f"[批量处理] 原始音频input_features形状: {input_features.shape}")
            
            # 预处理扰动音频
            perturbation_inputs = []
            for i in range(1, len(audios)):
                perturb_audio = audios[i]
                perturb_inputs = self.whisper_processor(perturb_audio, sampling_rate=16000, return_tensors="pt")
                perturb_input_features = perturb_inputs.input_features.to(self.device).half()
                perturbation_inputs.append(perturb_input_features)
                print(f"[批量处理] 扰动音频{i} input_features形状: {perturb_input_features.shape}")
            
            # 实现对比解码的LogitsProcessor
            class ContrastiveLogitsProcessor:
                def __init__(self, model, perturbation_inputs, perturbation_attention_masks, alpha=1.0, temperature=1.0, whisper_processor=None, perturb_context=None, device=None, repetition_penalty=1.2):
                    self.model = model
                    self.perturbation_inputs = perturbation_inputs
                    self.perturbation_attention_masks = perturbation_attention_masks
                    self.alpha = alpha
                    self.temperature = temperature
                    self.whisper_processor = whisper_processor
                    self.perturb_context = perturb_context
                    self.device = device if device else 'cuda'
                    self.repetition_penalty = repetition_penalty
                    self.generated_tokens = []

                def __call__(self, input_ids, logits):
                    # 计算扰动音频的logits
                    perturbation_logits = []
                    for i, perturb_input in enumerate(self.perturbation_inputs):
                        with torch.no_grad():
                            # 扰动使用任务+语言 token + 空格 padding，padding 长度与 input_ids 匹配
                            seq_len = input_ids.shape[1]
                            task_len = self.perturb_context.shape[1]
                            if seq_len > task_len:
                                padding_length = seq_len - task_len
                                padding = torch.full((1, padding_length), 220, dtype=torch.long, device=self.device)
                                current_perturb_context = torch.cat([self.perturb_context, padding], dim=1)
                            else:
                                current_perturb_context = self.perturb_context[:, :seq_len]
                            perturb_outputs = self.model(
                                input_features=perturb_input,
                                decoder_input_ids=current_perturb_context,
                                attention_mask=self.perturbation_attention_masks[i]
                            )
                            perturb_logits = perturb_outputs.logits[:, -1, :]
                            perturbation_logits.append(perturb_logits)
                    
                    # 执行对比解码
                    if perturbation_logits:
                        # 计算log-sum-exp
                        K = len(perturbation_logits)
                        
                        # 从第一个扰动logits获取设备信息
                        device = perturbation_logits[0].device
                        
                        # 计算所有扰动logits的log-sum-exp
                        sum_exp = torch.zeros_like(perturbation_logits[0], device=device, dtype=torch.float16)
                        max_logits = torch.max(torch.stack(perturbation_logits), dim=0)[0]
                        
                        # 计算exp并求和
                        for plogits in perturbation_logits:
                            exp_logits = torch.exp(plogits / self.temperature - max_logits / self.temperature)
                            sum_exp += exp_logits
                        
                        avg_exp = sum_exp / K
                        log_avg_exp = self.temperature * (torch.log(avg_exp) + max_logits / self.temperature)
                        
                        # 打印详细的计算过程
                        if hasattr(self, '_call_count'):
                            self._call_count += 1
                        else:
                            self._call_count = 1

                        if self._call_count % 1 == 0:
                            top_k = 5
                            perturb_names = ["Noise", "Silence", "TimeShift"]

                            def print_token_info(name, top_values, top_indices, tokens):
                                print(f"  {name} Top-{top_k}: {top_values.cpu().detach().numpy()}")
                                print(f"  {name} token IDs: {top_indices.cpu().tolist()}")
                                for idx, (tok_id, tok_text) in enumerate(zip(top_indices.cpu().tolist(), tokens)):
                                    char_info = [f"U+{ord(c):04X}" for c in tok_text] if tok_text else []
                                    print(f"    [{idx}] ID={tok_id}, chars={char_info}, text={repr(tok_text)}")
                                print(f"  {name} token: {tokens}")

                            original_top_values, original_top_indices = torch.topk(logits[0], top_k)
                            original_tokens = self.whisper_processor.batch_decode(original_top_indices.unsqueeze(0), skip_special_tokens=True)
                            print(f"\n[Token {self._call_count}] Contrastive Decoding Detail:")
                            print_token_info("Original", original_top_values, original_top_indices, original_tokens)

                            for i, plogits in enumerate(perturbation_logits):
                                perturb_top_values, perturb_top_indices = torch.topk(plogits[0], top_k)
                                perturb_tokens = self.whisper_processor.batch_decode(perturb_top_indices.unsqueeze(0), skip_special_tokens=True)
                                print_token_info(perturb_names[i], perturb_top_values, perturb_top_indices, perturb_tokens)

                            log_avg_exp_top_values, log_avg_exp_top_indices = torch.topk(log_avg_exp[0], top_k)
                            log_avg_exp_tokens = self.whisper_processor.batch_decode(log_avg_exp_top_indices.unsqueeze(0), skip_special_tokens=True)
                            print_token_info("log_avg_exp", log_avg_exp_top_values, log_avg_exp_top_indices, log_avg_exp_tokens)

                            contrastive_logits = (1 + self.alpha) * logits - self.alpha * log_avg_exp
                            contrastive_top_values, contrastive_top_indices = torch.topk(contrastive_logits[0], top_k)
                            contrastive_tokens = self.whisper_processor.batch_decode(contrastive_top_indices.unsqueeze(0), skip_special_tokens=True)
                            print_token_info("Contrastive", contrastive_top_values, contrastive_top_indices, contrastive_tokens)
                            print(f"  Formula: contrastive_logits = (1 + {self.alpha}) * logits - {self.alpha} * log_avg_exp")

                        # 计算对比logits - 严格按照论文公式
                        contrastive_logits = (1 + self.alpha) * logits - self.alpha * log_avg_exp

                        # 应用重复惩罚，防止生成重复token
                        # 跟踪所有已生成的token并惩罚重复
                        if len(self.generated_tokens) >= 2 and self.repetition_penalty != 1.0:
                            last_token = self.generated_tokens[-1] if self.generated_tokens else None
                            if last_token is not None:
                                # 如果最后一个token已经出现过多次，大幅惩罚
                                token_count = self.generated_tokens.count(last_token)
                                if token_count >= 2:
                                    penalty = self.repetition_penalty ** (token_count - 1)
                                    if contrastive_logits[0, last_token] > 0:
                                        contrastive_logits[0, last_token] /= penalty
                                    else:
                                        contrastive_logits[0, last_token] *= penalty

                            # 检查最后2个token的组合是否重复
                            if len(self.generated_tokens) >= 4:
                                last_two = tuple(self.generated_tokens[-2:])
                                # 检查这个组合之前是否出现过
                                for i in range(len(self.generated_tokens) - 2):
                                    if tuple(self.generated_tokens[i:i+2]) == last_two:
                                        # 惩罚最后一个token
                                        if contrastive_logits[0, last_token] > 0:
                                            contrastive_logits[0, last_token] /= (self.repetition_penalty * 1.5)
                                        else:
                                            contrastive_logits[0, last_token] *= (self.repetition_penalty * 1.5)
                                        break

                        # 记录生成的token
                        generated_ids = input_ids[0].tolist()
                        # 只在首次调用时初始化，之后累积新生成的token
                        if not hasattr(self, '_first_call') or not self._first_call:
                            self.generated_tokens = generated_ids[1:]  # 跳过第一个（是prompt）
                            self._first_call = True
                        else:
                            # 只添加新生成的token
                            if len(generated_ids) > len(self.generated_tokens) + 1:  # +1 因为包含prompt
                                new_tokens = generated_ids[len(self.generated_tokens) + 1:]
                                self.generated_tokens.extend(new_tokens)

                        return contrastive_logits
                    else:
                        return logits
            
            # 创建对比解码的LogitsProcessor
            # 为扰动音频创建注意力掩码
            perturbation_attention_masks = []
            for perturb_input in perturbation_inputs:
                mask = torch.ones(perturb_input.shape[:2], device=self.device, dtype=torch.float16)
                perturbation_attention_masks.append(mask)

            # 获取原始音频的注意力掩码
            if 'attention_mask' in inputs:
                original_attention_mask = inputs.attention_mask.to(self.device).half()
            else:
                original_attention_mask = torch.ones(input_features.shape[:2], device=self.device, dtype=torch.float16)

            # 获取任务和语言 token 构建扰动的 context
            forced_decoder_ids = self.whisper_processor.get_decoder_prompt_ids(
                task="transcribe",
                language=language,
                no_timestamps=True
            )
            if isinstance(forced_decoder_ids, list) and len(forced_decoder_ids) > 0:
                if isinstance(forced_decoder_ids[0], tuple):
                    forced_decoder_ids = [t[1] for t in forced_decoder_ids]
            # 构建扰动的 context：任务+语言 tokens + 空格 padding
            task_lang_tokens = torch.tensor([forced_decoder_ids], dtype=torch.long, device=self.device)
            perturb_context = task_lang_tokens

            contrastive_processor = ContrastiveLogitsProcessor(
                self.whisper_model,
                perturbation_inputs,
                perturbation_attention_masks,
                alpha=self.alpha,
                temperature=self.temperature,
                whisper_processor=self.whisper_processor,
                perturb_context=perturb_context,
                device=self.device,
                repetition_penalty=1.2
            )
            
            # 生成转录结果 - 严格按照论文设置：贪婪解码，禁用温度回退
            with torch.no_grad():
                generate_kwargs = {
                    "input_features": input_features,
                    "language": language,
                    "task": "transcribe",
                    "max_new_tokens": 200,
                    "logits_processor": [contrastive_processor],
                    "num_beams": 1,  # 贪婪解码
                    "temperature": 0.0,  # 禁用温度回退
                    "do_sample": False,  # 禁用采样
                    "attention_mask": original_attention_mask,
                    "repetition_penalty": 1.2,  # 添加重复惩罚
                    "no_repeat_ngram_size": 3,  # 防止3-gram重复
                }
                # 添加上下文提示
                if context:
                    try:
                        prompt_ids = self.whisper_processor.get_prompt_ids(context, return_tensors="pt")
                        if hasattr(prompt_ids, 'input_ids'):
                            prompt_ids = prompt_ids.input_ids
                        prompt_ids = prompt_ids.to(self.device)
                        generate_kwargs["prompt_ids"] = prompt_ids
                        print(f"[批量处理] 使用上下文提示，长度: {prompt_ids.shape[0] if hasattr(prompt_ids, 'shape') else len(prompt_ids)}")
                    except Exception as e:
                        print(f"[批量处理] 无法获取上下文提示 IDs: {e}")
                print("[批量处理] 开始生成转录结果...")
                outputs = self.whisper_model.generate(**generate_kwargs)
                print(f"[批量处理] 生成完成，outputs类型: {type(outputs)}")
            
            # 解码
            # 检查outputs是否是Tensor
            if isinstance(outputs, torch.Tensor):
                transcription = self.whisper_processor.batch_decode(outputs, skip_special_tokens=True)[0]
            else:
                transcription = self.whisper_processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0]
            
            # 过滤掉 tokenizer 词汇表中的无效字符 (U+FFFD - Replacement Character)
            transcription = transcription.replace('\ufffd', '')
            
            print(f"[批量处理] 转录结果: {transcription[:100]}...")
            
            # 构建segments和info
            segments = [{"text": transcription, "start": 0, "end": len(original_audio)/sr}]
            info = type('Info', (), {"language": language})()
            
            results.append((segments, info, None))
            
            # 为其他音频添加空结果（只需要原始音频的结果）
            for _ in range(len(audios) - 1):
                results.append(([], None, None))
        except Exception as e:
            print(f"处理音频时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            # 出错时添加空结果
            for _ in audios:
                results.append(([], None, None))
        
        # 更彻底的内存清理
        import gc
        # 清理缓存
        torch.cuda.empty_cache()
        # 强制垃圾回收
        gc.collect()
        # 再次清理缓存
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
        
        # 加载原始音频
        original_audio, sr = self._load_audio(audio_path)
        audio_duration = len(original_audio) / sr
        
        # 如果设置了最大处理时长且大于0，只处理前 max_duration 秒
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
        
        # 处理VAD检测到的语音片段
        processed_segments = []  # 已保留的片段，用作上下文
        original_segment_count = 0  # 原始片段总数
        temp_files = []
        
        try:
            if progress_callback:
                progress_callback(40, "开始按VAD片段处理...")
            
            # 逐个处理VAD检测到的语音片段
            total_vad_segments = len(self.speech_segments)
            if total_vad_segments == 0:
                print("[VAD] 未检测到语音活动，使用整个音频按30秒分块处理")
                # 使用整个音频，按30秒分块
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

                # 提取当前VAD片段的音频
                start_sample = int(start_time * sr)
                end_sample = int(end_time * sr)
                segment_audio = original_audio[start_sample:end_sample]
                
                # 生成扰动音频
                perturbations = self._generate_perturbations(segment_audio, sr)
                
                # 构建上下文（使用最近 N 个片段作为上下文）
                context = ""
                
                # 准备所有音频数据（原始音频 + 三种扰动）
                all_audios = [segment_audio] + perturbations
                
                # 批量处理当前VAD片段的所有音频（内存中直接处理）
                results = self._batch_process(all_audios, sr, language, context)
                
                # 提取原始片段的结果
                original_segments, original_info, original_audio_path = results[0]
                original_segments_list = list(original_segments)
                
                # 从original_info中获取语言信息
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
                    # 提取扰动音频的结果
                    perturbation_results = results[1:]
                
                    # 对当前VAD片段内的每个识别片段进行处理
                    for seg_idx, segment in enumerate(original_segments_list):
                        # 调试：打印原始时间戳
                        orig_start = segment.get('start', 0) if isinstance(segment, dict) else getattr(segment, 'start', 0)
                        orig_end = segment.get('end', 0) if isinstance(segment, dict) else getattr(segment, 'end', 0)

                        # 直接使用 VAD 的时间戳，而不是 Whisper 返回的时间戳
                        if isinstance(segment, dict):
                            segment['start'] = start_time
                            segment['end'] = end_time
                        else:
                            segment.start = start_time
                            segment.end = end_time

                        processed_segments.append(segment)
                        print(f"[对比解码] 片段 {i+1}-{seg_idx+1}: 原始ts={orig_start:.2f}-{orig_end:.2f} -> VAD ts={start_time:.2f}-{end_time:.2f}")
                
                # 清理临时文件
                for result in results:
                    _, _, temp_path = result
                    if temp_path and os.path.exists(temp_path):
                        try:
                            os.remove(temp_path)
                        except:
                            pass
            
            if progress_callback:
                progress_callback(100, "处理完成")
            
            # 构建结果
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

            # 注意：标点还原在 speech_recognizer.py 中进行，这里不再重复处理

            # 执行强制对齐（如果启用）
            if self.enable_alignment:
                print("\n[强制对齐] 开始执行强制对齐...")
                
                # 检查增强版语音识别器
                print(f"[强制对齐] 增强版语音识别器已初始化: {self.enhanced_recognizer is not None}")
                
                # 加载对齐模型
                detected_language = result.get('language', language)
                print(f"[强制对齐] 检测到语言: {detected_language}, 输入语言: {language}")
                
                if detected_language:
                    # 语言代码映射
                    lang_map = {
                        "japanese": "ja",
                        "chinese": "zh",
                        "english": "en",
                        "korean": "ko",
                    }
                    lang_key = detected_language.lower()
                    language_code = lang_map.get(lang_key, lang_key.split('-')[0][:2])  # 提取语言代码
                    print(f"[强制对齐] 语言代码: {language_code}")
                    
                    # 加载对齐模型
                    aligner = self.enhanced_recognizer.aligner
                    print(f"[强制对齐] 对齐器已初始化: {aligner is not None}")
                    
                    if aligner:
                        print(f"[强制对齐] 开始加载对齐模型...")
                        aligner.load_alignment_model(language_code)
                        print(f"[强制对齐] 对齐模型加载完成: {aligner.align_model is not None}")
                        
                        # 执行强制对齐
                        if aligner.align_model:
                            print(f"[强制对齐] 开始执行对齐...")
                            # 为了对齐，需要先保存临时音频文件
                            import tempfile
                            temp_audio_path = tempfile.mktemp(suffix=".wav")
                            print(f"[强制对齐] 临时音频文件: {temp_audio_path}")
                            
                            # 保存音频
                            print(f"[强制对齐] 保存音频，长度: {len(original_audio)} 采样点, 采样率: {sr}")
                            sf.write(temp_audio_path, original_audio, sr)
                            print(f"[强制对齐] 音频保存完成")
                            
                            try:
                                # 执行对齐，使用return_char_alignments获取字符级时间戳
                                print(f"[强制对齐] 执行对齐，段落数: {len(result['segments'])}")
                                aligned_segments = aligner.align(
                                    result['segments'],
                                    temp_audio_path,
                                    return_char_alignments=True
                                )
                                print(f"[强制对齐] 对齐完成，对齐后段落数: {len(aligned_segments)}")

                                # 使用对齐后的片段，但保留VAD的时间戳作为片段级别的时间戳
                                # 因为对齐后的片段已经包含了chars和words信息（词级时间戳）
                                # 需要确保每个片段都有chars和words信息
                                for i, aligned_seg in enumerate(aligned_segments):
                                    # 确保片段包含必要的时间戳信息
                                    if 'start' not in aligned_seg or 'end' not in aligned_seg:
                                        aligned_seg['start'] = aligned_seg.get('start', 0.0)
                                        aligned_seg['end'] = aligned_seg.get('end', 0.0)
                                    
                                    # 检查是否有chars信息
                                    if 'chars' not in aligned_seg or not aligned_seg['chars']:
                                        print(f"[强制对齐] 警告：片段 {i+1} 没有chars信息")
                                    
                                    # 检查是否有words信息
                                    if 'words' not in aligned_seg or not aligned_seg['words']:
                                        print(f"[强制对齐] 警告：片段 {i+1} 没有words信息")
                                
                                result['segments'] = aligned_segments
                                print(f"[强制对齐] 使用对齐后的片段（包含词级时间戳）")
                            except Exception as e:
                                print(f"[强制对齐] 对齐失败: {str(e)}")
                                print(f"[强制对齐] 回退到标点断句后的时间戳")
                                result['segments'] = pre_alignment_segments
                            finally:
                                # 清理临时文件
                                if os.path.exists(temp_audio_path):
                                    os.remove(temp_audio_path)
                                    print(f"[强制对齐] 临时文件已清理")
                    else:
                        print("[强制对齐] 对齐器未初始化")
                else:
                    print("[强制对齐] 未检测到语言")
            else:
                print("[强制对齐] 强制对齐功能已禁用")
            
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
            
            # 清理对齐模型
            if hasattr(self, 'aligner') and self.aligner:
                self.aligner.cleanup()

    def transcribe(self, audio_path: str, language: Optional[str] = "japanese", 
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
    
    # 初始化处理器
    processor = WhisperCDOriginal(
        args.model,
        alpha=args.alpha,
        temperature=args.temperature,
        snr_db=args.snr_db,
        temporal_shift=args.temporal_shift
    )
    
    # 定义进度回调
    def progress_callback(progress, message):
        print(f"进度: {progress}% - {message}")
    
    # 执行转录
    print(f"正在处理音频: {args.audio_path}")
    result = processor.transcribe(args.audio_path, args.language, progress_callback)
    
    # 打印结果
    print(f"检测到语言: {result['language']} (概率: {result['language_probability']:.2f})")
    print("识别结果:")
    for i, segment in enumerate(result['segments']):
        print(f"[{i+1}] {segment.start:.2f}s - {segment.end:.2f}s: {segment.text}")
