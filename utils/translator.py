# -*- coding: utf-8 -*-


import os
import gc
import re
import time

try:
    import torch
except ImportError:
    torch = None

from utils.llama_server_manager import LlamaServerManager
from utils.language_ratio_detector import check_translation_success, is_translation_valid


from config import MODEL_CACHE_DIR, config, TransParams, ServerParams


_active_server_manager = None

def clear_translator_cache(server_manager=None):
    """清空翻译模型缓存以释放内存并停止服务器进程"""
    global _active_server_manager
    target = server_manager or _active_server_manager
    try:
        if target is not None:
            target.stop_server()
            print("[llama-server] 已通过 server_manager 停止服务器进程")
    except Exception as e:
        print(f"[llama-server] 停止服务器时出错: {e}")

    _active_server_manager = None
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[内存管理] 已执行垃圾回收和显存清理")


class LlamaCppTranslator:
    """使用 llama-server HTTP API 运行 GGUF 模型的翻译器"""

    def __init__(self, model_path=None, server_params: ServerParams = None):
        self.system_prompt = "You are a translator. Your only task is to translate the given text from the source language to the target language. Output only the translation, nothing else. Do not include any instructions or explanations in your response."
        if server_params is None:
            server_params = ServerParams.from_dict(config.get_all())
        self.server_manager = LlamaServerManager(server_params=server_params)
        self.chat_history = []
        
        print(f"[llama-server翻译] 使用 HTTP API: {self.server_manager.host}:{self.server_manager.port}")
        print(f"[llama-server翻译] 模型: {self.server_manager.model_path}")

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
        
        sequences.sort(key=lambda x: x[2])
        
        result = text
        offset = 0
        for seq, repeat_count, start_pos in sequences:
            keep_seq = seq * keep_count
            original_len = len(seq) * repeat_count
            actual_pos = start_pos + offset
            if actual_pos >= len(result):
                continue
            max_len = len(result) - actual_pos
            if max_len <= 0:
                continue
            original_len = min(original_len, max_len)
            result = result[:actual_pos] + keep_seq + result[actual_pos + original_len:]
            offset += len(keep_seq) - original_len
        
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
        
        # Step 3.5: Remove spaces between CJK characters
        cjk_chars = r'\u3000-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af\uff00-\uffef\u3000-\u303f'
        text = re.sub(f'(?<=[{cjk_chars}])\\s+(?=[{cjk_chars}])', '', text)
        
        # Step 4: Compress repeated character sequences (existing logic)
        text = self.compress_repeated_sequences(text, keep_count)
        
        return text.strip()

    def translate(self, text, source_lang="en", target_lang="zh", trans_params=None):
        """翻译单个文本"""
        result = self._translate_multi_fallback(text, source_lang, target_lang, trans_params)
        return result[0] if isinstance(result, tuple) else result

    def _translate_multi_fallback(self, text, source_lang="en", target_lang="zh", trans_params=None, context=None):
        """使用 Chat API 进行翻译，保留翻译历史以利用 KV cache"""
        if trans_params is None:
            trans_params = TransParams()
        processed_text = self.preprocess_text(text)
        
        lang_map = {
            "zh": "Chinese", "en": "English", "ja": "Japanese", "ko": "Korean",
            "fr": "French", "de": "German", "es": "Spanish", "ru": "Russian",
            "ar": "Arabic", "hi": "Hindi", "pt": "Portuguese", "it": "Italian",
            "nl": "Dutch", "pl": "Polish"
        }

        source_lang_name = lang_map.get(source_lang, source_lang)
        target_lang_name = lang_map.get(target_lang, target_lang)

        if context:
            user_content = f"Context: {context}\n\nTranslate the following {source_lang_name} text to {target_lang_name}. Only output the translation, nothing else.\n\n{source_lang_name}: {processed_text}"
        else:
            user_content = f"Translate the following {source_lang_name} text to {target_lang_name}. Only output the translation, nothing else.\n\n{source_lang_name}: {processed_text}"

        try:
            temperature = trans_params.temperature
            top_k = trans_params.top_k
            top_p = trans_params.top_p
            repetition_penalty = trans_params.rep_penalty

            if not self.server_manager.ensure_server_running():
                raise RuntimeError("llama-server 启动失败，无法进行翻译")
            text_len = len(processed_text)
            if text_len <= 5:
                n_predict = 64
            elif text_len <= 20:
                n_predict = max(32, text_len * 2)
            else:
                n_predict = max(64, text_len * 2)
            max_context_tokens = getattr(self.server_manager, 'context_size', 4096)
            estimated_prompt_tokens = len(user_content) * 2 + sum(len(m['content']) * 2 for m in self.chat_history)
            # 如果估算的prompt token数超过max_context_tokens，截断chat_history
            if estimated_prompt_tokens > max_context_tokens:
                while self.chat_history and estimated_prompt_tokens > max_context_tokens:
                    removed = self.chat_history.pop(0)
                    estimated_prompt_tokens -= len(removed['content']) * 2
            max_n_predict = max_context_tokens - estimated_prompt_tokens - 200
            if max_n_predict > 0:
                n_predict = min(n_predict, max_n_predict)
            else:
                n_predict = min(n_predict, 256)
            # 使用 trans_params.max_output_tokens 限制最大输出
            n_predict = min(n_predict, trans_params.max_output_tokens)
            n_predict = max(n_predict, 64)

            messages = [{"role": "system", "content": self.system_prompt}]
            messages.extend(self.chat_history)
            messages.append({"role": "user", "content": user_content})

            output = self.server_manager.send_chat_request(
                messages,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repeat_penalty=repetition_penalty,
                n_predict=n_predict,
                timeout=300
            )
            
            if output is None:
                raise RuntimeError("llama-server 请求返回为空")

            if output == "":
                print(f"[翻译] 警告：翻译返回为空，原文: {processed_text[:50]}")

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

            return translation, processed_text

        except Exception as e:
            raise RuntimeError(f"llama-server 翻译失败: {str(e)}") from e

    def translate_batch(self, segments, source_lang="en", target_lang="zh", progress_callback=None, trans_params: TransParams = None):
        if trans_params is None:
            trans_params = TransParams()
        
        lang_map = {
            "zh": "Chinese", "en": "English", "ja": "Japanese", "ko": "Korean",
            "fr": "French", "de": "German", "es": "Spanish", "ru": "Russian",
            "ar": "Arabic", "hi": "Hindi", "pt": "Portuguese", "it": "Italian",
            "nl": "Dutch", "pl": "Polish"
        }
        source_lang_name = lang_map.get(source_lang, source_lang)
        target_lang_name = lang_map.get(target_lang, target_lang)

        total_segments = len(segments)
        print(f"[llama-server翻译] 开始单条翻译，共 {total_segments} 个片段")
        print(f"[llama-server翻译] 最大输出: {trans_params.max_output_tokens} tokens")
        
        batch_start_time = time.time()
        
        # 预计算所有片段的上下文
        context_cache = []
        for i in range(total_segments):
            current_text = segments[i].get('text', '')
            if len(current_text) <= 5:
                context_cache.append("")
                continue
            ctx_parts = []
            for j in range(max(0, i - trans_params.seg_ctx_window), i):
                ctx_parts.append(f"Previous: {segments[j].get('text', '')}")
            for j in range(i + 1, min(len(segments), i + trans_params.seg_ctx_window + 1)):
                ctx_parts.append(f"Next: {segments[j].get('text', '')}")
            context_cache.append(" ".join(ctx_parts))
        
        reset_session = trans_params.reset_session
        if reset_session:
            print("[llama-server翻译] 重置会话状态，确保全新的翻译环境...")
            self.chat_history = []
            self.server_manager.reset_session()
        else:
            self.server_manager.ensure_server_running()
        
        processed_count = 0
        for i, seg in enumerate(segments):
            # 跳过空文本
            original_text = seg.get('text', '').strip()
            if not original_text:
                seg['translated'] = ''
                continue

            text = seg["text"]
            context = context_cache[i]
            
            max_retries = trans_params.max_retries
            retry_count = 0
            translation = None
            seg_start_time = time.time()
            
            while retry_count < max_retries:
                try:
                    translation, processed_text = self._translate_multi_fallback(text, source_lang, target_lang, trans_params, context)

                    is_valid = is_translation_valid(text, translation, source_lang, target_lang)[0]

                    if is_valid:
                        seg["translated"] = translation
                        seg["_validated"] = True
                        if len(processed_text) > 5:
                            user_content = f"{source_lang_name}: {processed_text}"
                            self.chat_history.append({"role": "user", "content": user_content})
                            self.chat_history.append({"role": "assistant", "content": translation})
                            max_history_pairs = trans_params.seg_ctx_window
                            max_history_messages = max_history_pairs * 2
                            if len(self.chat_history) > max_history_messages:
                                self.chat_history = self.chat_history[-max_history_messages:]
                    else:
                        seg["translated"] = text
                        seg["_validated"] = False
                    
                    seg_elapsed = time.time() - seg_start_time
                    print(f'[翻译] 第{i+1}/{total_segments}条 ({seg_elapsed:.1f}s): "{text[:30]}..." → "{translation[:30]}..."')
                    break
                except Exception as e:
                    retry_count += 1
                    self.server_manager.ensure_server_running()
                    print(f"[llama-server翻译] 翻译失败，第{retry_count}次重试: {str(e)}")
                    if retry_count >= max_retries:
                        seg["translated"] = text
                        print(f"[llama-server翻译] 多次重试失败，使用原文")
            
            processed_count += 1
            if progress_callback and total_segments > 0:
                progress_callback(int(processed_count / total_segments * 100))
        
        print(f"[llama-server翻译] 单条翻译完成，共翻译 {total_segments} 个片段")
        
        # 验证翻译结果（首次翻译已验证，直接统计）
        success_count = sum(1 for seg in segments if seg.get("_validated", False))
        untranslated_indices = [i for i, seg in enumerate(segments) if not seg.get("_validated", False) and seg.get('text', '').strip()]
        fail_count = len(untranslated_indices)

        print(f"[llama-server翻译] 验证完成: 成功 {success_count} 个, 失败 {fail_count} 个")
        
        if untranslated_indices:
            print(f"[llama-server翻译] 发现 {len(untranslated_indices)} 个片段未翻译或翻译失败，开始重新翻译...")
            
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
                        context = context_cache[idx]
                    
                    current_retries[idx] += 1
                    retry_count = current_retries[idx]
                    
                    if retry_count > max_total_retries:
                        segment["translated"] = text
                        print(f"[llama-server翻译] 片段 {idx+1} 达到最大重试次数 {max_total_retries}，使用原文")
                        continue
                    
                    print(f"[llama-server翻译] 重新翻译片段 {idx+1}/{total_segments} (第{retry_count}次): {text[:30]}...")
                    
                    retry_start_time = time.time()
                    try:
                        translation, processed_text = self._translate_multi_fallback(text, source_lang, target_lang, trans_params, context)

                        is_valid = is_translation_valid(text, translation, source_lang, target_lang)[0]

                        if is_valid:
                            segment["translated"] = translation
                            if len(processed_text) > 5:
                                user_content = f"{source_lang_name}: {processed_text}"
                                self.chat_history.append({"role": "user", "content": user_content})
                                self.chat_history.append({"role": "assistant", "content": translation})
                                max_history_pairs = trans_params.seg_ctx_window
                                max_history_messages = max_history_pairs * 2
                                if len(self.chat_history) > max_history_messages:
                                    self.chat_history = self.chat_history[-max_history_messages:]
                            retry_elapsed = time.time() - retry_start_time
                            print(f'[翻译] 第{idx+1}/{total_segments}条 ({retry_elapsed:.1f}s): "{text[:30]}..." → "{translation[:30]}..."')
                        else:
                            remaining_indices.append(idx)
                    except Exception as e:
                        self.server_manager.ensure_server_running()
                        print(f"[llama-server翻译] 重新翻译失败，第{retry_count}次重试: {str(e)}")
                        remaining_indices.append(idx)
            
            if remaining_indices:
                print(f"[llama-server翻译] 仍有 {len(remaining_indices)} 个片段翻译失败，使用原文")
                for idx in remaining_indices:
                    segments[idx]["translated"] = segments[idx]["text"]
            
            print(f"[llama-server翻译] 重新翻译完成")
        else:
            print(f"[llama-server翻译] 所有片段翻译成功，无需重新翻译")
        
        for seg in segments:
            seg.pop("_validated", None)
        
        batch_elapsed = time.time() - batch_start_time
        avg_time = batch_elapsed / total_segments if total_segments > 0 else 0
        print(f"[llama-server翻译] 翻译统计: 共 {total_segments} 条, 总耗时 {batch_elapsed:.1f}s, 平均 {avg_time:.2f}s/条")
        
        return segments


def get_local_model_path(model_path):
    """获取本地模型路径，复用 LlamaServerManager 的模型查找逻辑"""
    from utils.llama_server_manager import LlamaServerManager
    server_params = ServerParams.from_dict(config.get_all())
    manager = LlamaServerManager(server_params=server_params)
    if manager.model_path:
        print(f"[翻译] 找到本地模型: {manager.model_path}")
        return os.path.dirname(manager.model_path)
    return None


def translate_text(recognized_result, model_path, progress_callback=None,
                   target_language="zh", trans_params: TransParams = None, server_params: ServerParams = None):
    """翻译识别结果"""
    if trans_params is None:
        trans_params = TransParams()
    if not target_language:
        target_language = "zh"

    local_model_path = get_local_model_path(model_path)
    if not local_model_path:
        error_msg = f"本地模型不存在: {model_path}，请确保模型已在models目录中"
        print(f"[错误信息] {error_msg}")
        raise FileNotFoundError(error_msg)

    translated_result = translate_with_llama_server(recognized_result, progress_callback, target_language, trans_params, server_params)
    
    if 'segments' in translated_result:
        has_translation = any('translated' in seg for seg in translated_result['segments'])
        print(f"[翻译] 翻译完成，{len(translated_result['segments'])} 个片段，其中 {sum(1 for seg in translated_result['segments'] if 'translated' in seg)} 个片段有翻译")
    
    return translated_result


def translate_with_llama_server(recognized_result, progress_callback, target_language, trans_params=None, server_params=None):
    """使用 llama-server HTTP API 运行 GGUF 模型进行翻译"""
    if trans_params is None:
        trans_params = TransParams()
    
    source_language = recognized_result.get('language', 'auto')
    if source_language == 'auto':
        if recognized_result.get('text'):
            text = recognized_result['text']
            if any('\u3040' <= c <= '\u30ff' for c in text):  # Japanese kana (hiragana + katakana)
                source_language = 'ja'
            elif any('\uac00' <= c <= '\ud7af' for c in text):  # Korean
                source_language = 'ko'
            elif any('\u4e00' <= c <= '\u9fff' for c in text):  # CJK ideographs
                source_language = 'zh'
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
    
    server_params = server_params or ServerParams.from_dict(config.get_all())
    translator = LlamaCppTranslator(server_params=server_params)

    global _active_server_manager
    _active_server_manager = translator.server_manager

    try:
        translated_segments = translator.translate_batch(
            segments,
            source_lang=source_language,
            target_lang=target_language,
            progress_callback=progress_callback,
            trans_params=trans_params
        )
        
        recognized_result['segments'] = translated_segments
        
        # CJK语言不使用空格连接
        is_cjk_target = target_language and target_language.startswith(('zh', 'ja', 'ko'))
        separator = '' if is_cjk_target else ' '
        translated_text = separator.join([seg.get('translated', seg.get('text', '')) for seg in translated_segments])
        recognized_result['translated_text'] = translated_text
        
        return recognized_result
    finally:
        clear_translator_cache(translator.server_manager)
