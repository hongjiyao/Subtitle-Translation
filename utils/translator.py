# -*- coding: utf-8 -*-


import os
import gc
import re

from utils.llama_server_manager import LlamaServerManager
from utils.language_ratio_detector import check_translation_success, is_translation_valid
from utils.logger import timestamp_print

from config import MODEL_CACHE_DIR, config, TransParams


def clear_translator_cache(server_manager=None):
    """清空翻译模型缓存以释放内存并停止服务器进程"""
    try:
        if server_manager is not None:
            server_manager.stop_server()
            timestamp_print("[llama-server] 已通过 server_manager 停止服务器进程")
        else:
            import subprocess
            subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"], capture_output=True, text=True)
            timestamp_print("[llama-server] 已停止服务器进程")
    except Exception as e:
        timestamp_print(f"[llama-server] 停止服务器时出错: {e}")
    
    gc.collect()
    timestamp_print("[内存管理] 已执行垃圾回收")


class LlamaCppTranslator:
    """使用 llama-server HTTP API 运行 GGUF 模型的翻译器"""

    def __init__(self, model_path=None):
        system_prompt = "You are a translator. Your only task is to translate the given text from the source language to the target language. Output only the translation, nothing else. Do not include any instructions or explanations in your response."
        self.server_manager = LlamaServerManager(system_prompt=system_prompt, port=8080)
        
        timestamp_print(f"[llama-server翻译] 使用 HTTP API: {self.server_manager.host}:{self.server_manager.port}")
        timestamp_print(f"[llama-server翻译] 模型: {self.server_manager.model_path}")

    def compress_repeated_sequences(self, text: str, keep_count: int = 1) -> str:
        """
        压缩文本中的重复字符序列
        
        Args:
            text: 原始文本
            keep_count: 保留的重复次数（默认保留1个）
        
        Returns:
            压缩后的文本
        """
        if not text:
            return text
        
        sequences = []
        i = 0
        n = len(text)
        
        while i < n:
            if n - i >= 3:
                current_char = text[i]
                repeat_count = 1
                j = i + 1
                while j < n and text[j] == current_char:
                    repeat_count += 1
                    j += 1
                if repeat_count >= 3:
                    sequences.append((current_char, repeat_count, i))
                    i = j
                    continue
            
            found = False
            for seq_len in range(min(10, n - i), 1, -1):
                seq = text[i:i + seq_len]
                repeat_count = 1
                j = i + seq_len
                while j + seq_len <= n and text[j:j + seq_len] == seq:
                    repeat_count += 1
                    j += seq_len
                
                if repeat_count >= 3:
                    sequences.append((seq, repeat_count, i))
                    i = j
                    found = True
                    break
            
            if not found:
                i += 1
        
        if not sequences:
            return text
        
        sequences.sort(key=lambda x: x[2], reverse=True)
        
        result = text
        for seq, repeat_count, start_pos in sequences:
            keep_seq = seq * keep_count
            original_len = len(seq) * repeat_count
            max_len = len(result) - start_pos
            original_len = min(original_len, max_len)
            result = result[:start_pos] + keep_seq + result[start_pos + original_len:]
        
        cleaned = []
        i = 0
        n = len(result)
        while i < n:
            if i + 2 < n and result[i] == result[i+1] == result[i+2]:
                cleaned.append(result[i])
                i += 3
            else:
                cleaned.append(result[i])
                i += 1
        
        return ''.join(cleaned)

    def preprocess_text(self, text, keep_count=1):
        """预处理文本，保留语义标点，移除装饰性标点，压缩重复序列"""
        if not text or not text.strip():
            return ""
        
        text = text.strip()
        
        # Step 1: Compress consecutive repeated punctuation (2+ same punct → 1)
        text = re.sub(r'([。！？…？！；:，、.!?;:—–～♪★♡※~])\1+', r'\1', text)
        
        # Step 2: Remove decorative/non-semantic punctuation (NOT in whitelist)
        semantic_punct = set('。！？…？！；:，、.!?;:—–')
        result = []
        for ch in text:
            if ch in semantic_punct or ch.isalnum() or ch.isspace() or (ord(ch) > 0x3000 and ch not in '～♪★♡※○●◎◇◆□■△▽'):
                result.append(ch)
        text = ''.join(result)
        
        # Step 3: Collapse multiple whitespace to single space
        text = re.sub(r'\s+', ' ', text)
        
        # Step 4: Compress repeated character sequences (existing logic)
        text = self.compress_repeated_sequences(text, keep_count)
        
        return text.strip()

    def translate(self, text, source_lang="en", target_lang="zh", trans_params=None):
        """翻译单个文本"""
        return self._translate_multi_fallback(text, source_lang, target_lang, trans_params)

    def _translate_multi_fallback(self, text, source_lang="en", target_lang="zh", trans_params=None, context=None):
        """使用单次调用模式进行翻译"""
        if trans_params is None:
            trans_params = TransParams()
        processed_text = self.preprocess_text(text)
        
        lang_map = {
            "zh": "Chinese",
            "en": "English",
            "ja": "Japanese",
            "ko": "Korean",
            "fr": "French",
            "de": "German",
            "es": "Spanish",
            "ru": "Russian",
            "ar": "Arabic",
            "hi": "Hindi",
            "pt": "Portuguese",
            "it": "Italian",
            "nl": "Dutch",
            "pl": "Polish"
        }

        source_lang_name = lang_map.get(source_lang, source_lang)
        target_lang_name = lang_map.get(target_lang, target_lang)

        if context:
            prompt = f"Context: {context}\n\nTranslate the following {source_lang_name} text to {target_lang_name}. Only output the translation, nothing else.\n\n{source_lang_name}: {processed_text}\n{target_lang_name}:"
        else:
            prompt = f"Translate the following {source_lang_name} text to {target_lang_name}. Only output the translation, nothing else.\n\n{source_lang_name}: {processed_text}\n{target_lang_name}:"

        try:
            temperature = trans_params.temperature
            top_k = trans_params.top_k
            top_p = trans_params.top_p
            repetition_penalty = trans_params.rep_penalty
            
            self.server_manager.ensure_server_running()
            n_predict = max(100, len(processed_text) * 3)
            output = self.server_manager.send_request(
                prompt,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repeat_penalty=repetition_penalty,
                n_predict=n_predict,
                timeout=300
            )
            
            if output is None:
                raise RuntimeError("llama-server 请求返回为空")

            output = output.strip()
            
            special_tokens = [
                "<|startoftext|>", "</s>", "<|eos|>", 
                "<|extra_0|>", "<|extra_4|>", "<|pad|>",
                " [end of text]", "[end of text]"
            ]
            for token in special_tokens:
                output = output.replace(token, "")
            
            translation = output.strip()
            translation = " ".join(translation.split())
            
            return translation

        except Exception as e:
            raise RuntimeError(f"llama-server 翻译失败: {str(e)}")

    def translate_batch(self, segments, source_lang="en", target_lang="zh", progress_callback=None, trans_params: TransParams = None):
        if trans_params is None:
            trans_params = TransParams()
        
        def extract_context(segments, index, window_size=None):
            """提取上下文信息"""
            if window_size is None:
                window_size = trans_params.seg_ctx_window
            current_text = segments[index].get('text', '')
            if len(current_text) <= 5:
                return ""
            context = []
            for i in range(max(0, index - window_size), index):
                context.append(f"Previous: {segments[i].get('translated', segments[i]['text'])}")
            for i in range(index + 1, min(len(segments), index + window_size + 1)):
                context.append(f"Next: {segments[i]['text']}")
            return " ".join(context)
        
        total_segments = len(segments)
        timestamp_print(f"[llama-server翻译] 开始单条翻译，共 {total_segments} 个片段")
        timestamp_print(f"[llama-server翻译] 上下文大小: {trans_params.ctx_size}, 最大输出: {trans_params.max_output_tokens} tokens")
        
        reset_session = trans_params.reset_session
        if reset_session:
            timestamp_print("[llama-server翻译] 重置会话状态，确保全新的翻译环境...")
            self.server_manager.reset_session()
        else:
            self.server_manager.ensure_server_running()
        
        processed_count = 0
        for i, segment in enumerate(segments):
            text = segment["text"]
            context = extract_context(segments, i)
            
            max_retries = trans_params.max_retries
            retry_count = 0
            translation = None
            
            while retry_count < max_retries:
                try:
                    translation = self._translate_multi_fallback(text, source_lang, target_lang, trans_params, context)
                    
                    is_valid = is_translation_valid(text, translation, source_lang, target_lang)[0]
                    
                    if is_valid:
                        segment["translated"] = translation
                    else:
                        segment["translated"] = text
                    
                    timestamp_print(f'[翻译] 第{i+1}/{total_segments}条: "{text[:30]}..." → "{translation[:30]}..."')
                    break
                except Exception as e:
                    retry_count += 1
                    error_str = str(e).lower()
                    if 'connection' in error_str or 'connect' in error_str or 'refused' in error_str:
                        self.server_manager.reset_session()
                    else:
                        self.server_manager.ensure_server_running()
                    timestamp_print(f"[llama-server翻译] 翻译失败，第{retry_count}次重试: {str(e)}")
                    if retry_count >= max_retries:
                        segment["translated"] = text
                        timestamp_print(f"[llama-server翻译] 多次重试失败，使用原文")
            
            processed_count += 1
            if progress_callback and total_segments > 0:
                progress_callback(int(processed_count / total_segments * 100))
        
        timestamp_print(f"[llama-server翻译] 单条翻译完成，共翻译 {total_segments} 个片段")
        
        # 验证翻译结果
        timestamp_print(f"[llama-server翻译] 开始验证翻译结果...")
        
        success_count = 0
        fail_count = 0
        untranslated_indices = []
        
        for i, segment in enumerate(segments):
            original_text = segment["text"]
            translated_text = segment.get("translated", "")
            
            success, target_ratio, lang_counts = is_translation_valid(
                original_text, translated_text, source_lang, target_lang, threshold=0.5
            )
            
            if success:
                success_count += 1
            else:
                untranslated_indices.append(i)
                fail_count += 1
        
        timestamp_print(f"[llama-server翻译] 验证完成: 成功 {success_count} 个, 失败 {fail_count} 个")
        
        if untranslated_indices:
            timestamp_print(f"[llama-server翻译] 发现 {len(untranslated_indices)} 个片段未翻译或翻译失败，开始重新翻译...")
            
            max_total_retries = trans_params.max_total_retries
            current_retries = {idx: 0 for idx in untranslated_indices}
            remaining_indices = untranslated_indices.copy()
            
            while remaining_indices and max(current_retries.values()) < max_total_retries:
                current_batch = remaining_indices.copy()
                remaining_indices = []
                
                for idx in current_batch:
                    segment = segments[idx]
                    text = segment["text"]
                    if len(text) <= 5:
                        context = ""
                    else:
                        context = extract_context(segments, idx)
                    
                    current_retries[idx] += 1
                    retry_count = current_retries[idx]
                    
                    if retry_count > max_total_retries:
                        segment["translated"] = text
                        timestamp_print(f"[llama-server翻译] 片段 {idx+1} 达到最大重试次数 {max_total_retries}，使用原文")
                        continue
                    
                    timestamp_print(f"[llama-server翻译] 重新翻译片段 {idx+1}/{total_segments} (第{retry_count}次): {text[:30]}...")
                    
                    try:
                        translation = self._translate_multi_fallback(text, source_lang, target_lang, trans_params, context)
                        
                        is_valid = is_translation_valid(text, translation, source_lang, target_lang)[0]
                        
                        if is_valid:
                            segment["translated"] = translation
                            timestamp_print(f'[翻译] 第{idx+1}/{total_segments}条: "{text[:30]}..." → "{translation[:30]}..."')
                        else:
                            remaining_indices.append(idx)
                    except Exception as e:
                        error_str = str(e).lower()
                        if 'connection' in error_str or 'connect' in error_str or 'refused' in error_str:
                            self.server_manager.reset_session()
                        else:
                            self.server_manager.ensure_server_running()
                        timestamp_print(f"[llama-server翻译] 重新翻译失败，第{retry_count}次重试: {str(e)}")
                        remaining_indices.append(idx)
            
            if remaining_indices:
                timestamp_print(f"[llama-server翻译] 仍有 {len(remaining_indices)} 个片段翻译失败，使用原文")
                for idx in remaining_indices:
                    segments[idx]["translated"] = segments[idx]["text"]
            
            timestamp_print(f"[llama-server翻译] 重新翻译完成")
        else:
            timestamp_print(f"[llama-server翻译] 所有片段翻译成功，无需重新翻译")
        
        return segments


def get_local_model_path(model_path):
    """获取本地模型路径"""
    model_name = model_path.replace("tencent/", "").replace("tencent--", "").split("/")[-1]

    possible_paths = [
        os.path.join(MODEL_CACHE_DIR, model_name),
        os.path.join(MODEL_CACHE_DIR, f"tencent--{model_name}"),
        os.path.join(MODEL_CACHE_DIR, "tencent", model_name),
        os.path.join(MODEL_CACHE_DIR, "tencent--HY-MT1.5-7B-GGUF"),
        os.path.join(MODEL_CACHE_DIR, "HY-MT1.5-7B-GGUF"),
    ]

    for path in possible_paths:
        if os.path.exists(path) and os.path.isdir(path):
            timestamp_print(f"[翻译] 找到本地模型: {path}")
            return path

    direct_path = os.path.join(MODEL_CACHE_DIR, model_path.replace("/", "--"))
    if os.path.exists(direct_path) and os.path.isdir(direct_path):
        timestamp_print(f"[翻译] 找到本地模型: {direct_path}")
        return direct_path

    return None


def translate_text(recognized_result, model_path, device_choice="auto", progress_callback=None,
                   beam_size=1, max_length=256, target_language="zh", trans_params: TransParams = None):
    """翻译识别结果"""
    if trans_params is None:
        trans_params = TransParams()
    if not target_language:
        target_language = "zh"

    local_model_path = get_local_model_path(model_path)
    if not local_model_path:
        error_msg = f"本地模型不存在: {model_path}，请确保模型已在models目录中"
        timestamp_print(f"[错误信息] {error_msg}")
        raise FileNotFoundError(error_msg)

    translated_result = translate_with_llama_server(recognized_result, progress_callback, target_language, trans_params)
    
    if 'segments' in translated_result:
        has_translation = any('translated' in seg for seg in translated_result['segments'])
        timestamp_print(f"[翻译] 翻译完成，{len(translated_result['segments'])} 个片段，其中 {sum(1 for seg in translated_result['segments'] if 'translated' in seg)} 个片段有翻译")
    
    return translated_result


def translate_with_llama_server(recognized_result, progress_callback, target_language, trans_params=None):
    """使用 llama-server HTTP API 运行 GGUF 模型进行翻译"""
    if trans_params is None:
        trans_params = TransParams()
    
    source_language = recognized_result.get('language', 'auto')
    if source_language == 'auto':
        if recognized_result.get('text'):
            text = recognized_result['text']
            if any('\u4e00' <= c <= '\u9fff' for c in text):
                source_language = 'zh'
            elif any('\u3040' <= c <= '\u30ff' for c in text):
                source_language = 'ja'
            elif any('\uac00' <= c <= '\ud7af' for c in text):
                source_language = 'ko'
            else:
                source_language = 'en'
        else:
            source_language = 'en'
    
    segments = recognized_result.get('segments', [])
    if not segments and recognized_result.get('text'):
        segments = [{
            'id': 0,
            'text': recognized_result['text'],
            'start': 0,
            'end': 0
        }]
    
    for seg in segments:
        if 'text' in seg:
            original_text = seg['text']
            seg['original_text'] = original_text
    
    translator = LlamaCppTranslator()
    
    try:
        translated_segments = translator.translate_batch(
            segments,
            source_lang=source_language,
            target_lang=target_language,
            progress_callback=progress_callback,
            trans_params=trans_params
        )
        
        recognized_result['segments'] = translated_segments
        
        translated_text = ''.join([seg.get('translated', seg.get('text', '')) for seg in translated_segments])
        recognized_result['translated_text'] = translated_text
        
        return recognized_result
    finally:
        clear_translator_cache(translator.server_manager)
