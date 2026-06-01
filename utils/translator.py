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


def clear_translator_cache(server_manager=None):
    """清空翻译模型缓存以释放内存并停止服务器进程"""
    try:
        if server_manager is not None:
            server_manager.stop_server()
            print("[llama-server] 已通过 server_manager 停止服务器进程")
    except Exception as e:
        print(f"[llama-server] 停止服务器时出错: {e}")

    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[内存管理] 已执行垃圾回收和显存清理")


_TOKEN_ESTIMATOR_CONFIGS = {
    "qwen2": {"cjk_ratio": 1.5, "latin_ratio": 0.3, "punct_ratio": 0.5},
    "default": {"cjk_ratio": 2.0, "latin_ratio": 0.25, "punct_ratio": 0.5},
}

_CJK_PUNCT = set('。！？、；：""''【】《》…—・～♪★♡※○●◎◇◆□■△▽')


def _estimate_token_count(text, model_family="default"):
    cfg = _TOKEN_ESTIMATOR_CONFIGS.get(model_family, _TOKEN_ESTIMATOR_CONFIGS["default"])
    cjk_count = 0
    punct_count = 0
    for c in text:
        if '\u4e00' <= c <= '\u9fff' or '\u3040' <= c <= '\u30ff' or '\uac00' <= c <= '\ud7af':
            cjk_count += 1
        elif c in _CJK_PUNCT:
            punct_count += 1
    latin_count = len(text) - cjk_count - punct_count
    return int(cjk_count * cfg["cjk_ratio"] + latin_count * cfg["latin_ratio"] + punct_count * cfg["punct_ratio"])


_ALLOWED_LANGUAGES = {
    'zh': 'Chinese', 'chinese': 'Chinese', '中文': 'Chinese',
    'en': 'English', 'english': 'English', '英语': 'English',
    'ja': 'Japanese', 'japanese': 'Japanese', '日本語': 'Japanese',
    'ko': 'Korean', 'korean': 'Korean', '韩语': 'Korean',
    'fr': 'French', 'french': 'French', '法语': 'French',
    'de': 'German', 'german': 'German', '德语': 'German',
    'es': 'Spanish', 'spanish': 'Spanish', '西班牙语': 'Spanish',
    'ru': 'Russian', 'russian': 'Russian', '俄语': 'Russian',
    'ar': 'Arabic', 'arabic': 'Arabic', '阿拉伯语': 'Arabic',
    'hi': 'Hindi', 'hindi': 'Hindi', '印地语': 'Hindi',
    'pt': 'Portuguese', 'portuguese': 'Portuguese', '葡萄牙语': 'Portuguese',
    'it': 'Italian', 'italian': 'Italian', '意大利语': 'Italian',
    'nl': 'Dutch', 'dutch': 'Dutch', '荷兰语': 'Dutch',
    'pl': 'Polish', 'polish': 'Polish', '波兰语': 'Polish',
}

_LANG_MAP = {k: v for k, v in _ALLOWED_LANGUAGES.items() if len(k) == 2}


def _sanitize_language(lang_input, default='English'):
    if not lang_input:
        return default
    lang_lower = lang_input.lower().strip()
    if lang_lower in _ALLOWED_LANGUAGES:
        return _ALLOWED_LANGUAGES[lang_lower]
    return default


class LlamaCppTranslator:
    """使用 llama-server HTTP API 运行 GGUF 模型的翻译器"""

    def __init__(self, model_path=None, server_params: ServerParams = None):
        self.system_prompt = "You are a translator. Your only task is to translate the given text from the source language to the target language. Output only the translation, nothing else. Do not include any instructions or explanations in your response."
        if server_params is None:
            server_params = ServerParams.from_dict(config.get_all())
        self._server_manager = LlamaServerManager(server_params=server_params)
        self.chat_history = []
        translator = config.get('translator', 'tencent/HY-MT1.5-1.8B-GGUF-Q8_0')
        self._model_family = "qwen2" if "qwen" in translator.lower() else "default"
        
        print(f"[llama-server翻译] 使用 HTTP API: {self._server_manager.host}:{self._server_manager.port}")
        print(f"[llama-server翻译] 模型: {self._server_manager.model_path}")

    def compress_repeated_sequences(self, text: str, keep_count: int = 1) -> str:
        if not text:
            return text

        result = re.sub(r'(.)\1{2,}', lambda m: m.group(1) * keep_count, text)

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
        
        source_lang_name = _sanitize_language(source_lang, default='English')
        target_lang_name = _sanitize_language(target_lang, default='Chinese')

        if context:
            user_content = f"Context: {context}\n\nTranslate the following {source_lang_name} text to {target_lang_name}. Only output the translation, nothing else.\n\n{source_lang_name}: {processed_text}"
        else:
            user_content = f"Translate the following {source_lang_name} text to {target_lang_name}. Only output the translation, nothing else.\n\n{source_lang_name}: {processed_text}"

        try:
            temperature = trans_params.temperature
            top_k = trans_params.top_k
            top_p = trans_params.top_p
            repetition_penalty = trans_params.rep_penalty

            if not self._server_manager.ensure_server_running():
                raise RuntimeError("llama-server 启动失败，无法进行翻译")
            text_len = len(processed_text)
            if text_len <= trans_params.short_text_threshold:
                n_predict = 64
            elif text_len <= 20:
                n_predict = max(32, text_len * 2)
            else:
                n_predict = max(64, text_len * 2)
            max_context_tokens = getattr(self._server_manager, 'context_size', 4096)
            system_prompt_tokens = _estimate_token_count(self.system_prompt, self._model_family)
            estimated_prompt_tokens = system_prompt_tokens + _estimate_token_count(user_content, self._model_family) + sum(_estimate_token_count(m['content'], self._model_family) for m in self.chat_history)
            if estimated_prompt_tokens > max_context_tokens:
                while self.chat_history and estimated_prompt_tokens > max_context_tokens:
                    removed = self.chat_history.pop(0)
                    estimated_prompt_tokens -= _estimate_token_count(removed['content'])
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

            output = self._server_manager.send_chat_request(
                messages,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repeat_penalty=repetition_penalty,
                n_predict=n_predict,
                timeout=trans_params.request_timeout
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

    def _translate_and_validate(self, text, source_lang, target_lang, trans_params, context, source_lang_name):
        """翻译并验证单个片段，验证通过时更新聊天历史"""
        translation, processed_text = self._translate_multi_fallback(text, source_lang, target_lang, trans_params, context)
        is_valid = is_translation_valid(text, translation, source_lang, target_lang, trans_params=trans_params)[0]

        if is_valid:
            if processed_text.strip():
                user_content = f"{source_lang_name}: {processed_text}"
                self.chat_history.append({"role": "user", "content": user_content})
                self.chat_history.append({"role": "assistant", "content": translation})
                max_history_pairs = trans_params.seg_ctx_window
                max_history_messages = max_history_pairs * 2
                if len(self.chat_history) > max_history_messages:
                    self.chat_history = self.chat_history[-max_history_messages:]

        return translation, processed_text, is_valid

    def _translate_initial(self, segments, source_lang, target_lang, trans_params, progress_callback):
        total_segments = len(segments)
        source_lang_name = _sanitize_language(source_lang, default='English')

        context_cache = []
        max_ctx_tokens = trans_params.max_context_tokens
        for i in range(total_segments):
            current_text = segments[i].get('text', '')
            if not current_text.strip() or max_ctx_tokens <= 0:
                context_cache.append("")
                continue
            ctx_parts = []
            ctx_token_count = 0
            for j in range(max(0, i - trans_params.seg_ctx_window), i):
                part = f"Previous: {segments[j].get('text', '')}"
                part_tokens = _estimate_token_count(part, self._model_family)
                if ctx_token_count + part_tokens > max_ctx_tokens:
                    break
                ctx_parts.insert(0, part)
                ctx_token_count += part_tokens
            for j in range(i + 1, min(len(segments), i + trans_params.seg_ctx_window + 1)):
                part = f"Next: {segments[j].get('text', '')}"
                part_tokens = _estimate_token_count(part, self._model_family)
                if ctx_token_count + part_tokens > max_ctx_tokens:
                    break
                ctx_parts.append(part)
                ctx_token_count += part_tokens
            context_cache.append(" ".join(ctx_parts))

        reset_session = trans_params.reset_session
        if reset_session:
            print("[llama-server翻译] 重置会话状态，确保全新的翻译环境...")
            self.chat_history = []
            self._server_manager.reset_session()
        else:
            self._server_manager.ensure_server_running()

        processed_count = 0
        for i, seg in enumerate(segments):
            original_text = seg.get('text', '').strip()
            if not original_text:
                seg['translated'] = ''
                continue

            text = seg.get("text", "")
            context = context_cache[i]
            max_retries = trans_params.max_retries
            retry_count = 0
            translation = None
            seg_start_time = time.time()

            while retry_count < max_retries:
                try:
                    translation, processed_text, is_valid = self._translate_and_validate(
                        text, source_lang, target_lang, trans_params, context, source_lang_name
                    )

                    if is_valid:
                        seg["translated"] = translation
                        seg["_validated"] = True
                    else:
                        seg["translated"] = text
                        seg["_validated"] = False

                    seg_elapsed = time.time() - seg_start_time
                    print(f'[翻译] 第{i+1}/{total_segments}条 ({seg_elapsed:.1f}s): "{text[:30]}..." → "{translation[:30]}..."')
                    break
                except Exception as e:
                    retry_count += 1
                    self._server_manager.ensure_server_running()
                    print(f"[llama-server翻译] 翻译失败，第{retry_count}次重试: {str(e)}")
                    if retry_count >= max_retries:
                        seg["translated"] = text
                        print(f"[llama-server翻译] 多次重试失败，使用原文")

            processed_count += 1
            if progress_callback and total_segments > 0:
                progress_callback(int(processed_count / total_segments * 100))

        success_count = sum(1 for seg in segments if seg.get("_validated", False))
        untranslated_indices = [i for i, seg in enumerate(segments) if not seg.get("_validated", False) and seg.get('text', '').strip()]
        return segments, success_count, untranslated_indices, context_cache

    def _validate_translations(self, segments, source_lang, target_lang):
        success_count = sum(1 for seg in segments if seg.get("_validated", False))
        failed_indices = [i for i, seg in enumerate(segments) if not seg.get("_validated", False) and seg.get('text', '').strip()]
        fail_count = len(failed_indices)
        print(f"[llama-server翻译] 验证完成: 成功 {success_count} 个, 失败 {fail_count} 个")
        return segments, failed_indices

    def _retry_untranslated(self, segments, remaining_indices, source_lang, target_lang, trans_params, context_cache):
        total_segments = len(segments)
        source_lang_name = _sanitize_language(source_lang, default='English')
        max_total_retries = trans_params.max_total_retries
        current_retries = {idx: 0 for idx in remaining_indices}

        while remaining_indices and max(current_retries.values()) < max_total_retries:
            current_batch = remaining_indices.copy()
            remaining_indices = []

            for idx in current_batch:
                segment = segments[idx]
                text = segment.get("text", "")
                if len(text) <= trans_params.short_text_threshold:
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
                    translation, processed_text, is_valid = self._translate_and_validate(
                        text, source_lang, target_lang, trans_params, context, source_lang_name
                    )

                    if is_valid:
                        segment["translated"] = translation
                        retry_elapsed = time.time() - retry_start_time
                        print(f'[翻译] 第{idx+1}/{total_segments}条 ({retry_elapsed:.1f}s): "{text[:30]}..." → "{translation[:30]}..."')
                    else:
                        remaining_indices.append(idx)
                except Exception as e:
                    self._server_manager.ensure_server_running()
                    print(f"[llama-server翻译] 重新翻译失败，第{retry_count}次重试: {str(e)}")
                    remaining_indices.append(idx)

        if remaining_indices:
            print(f"[llama-server翻译] 仍有 {len(remaining_indices)} 个片段翻译失败，使用原文")
            for idx in remaining_indices:
                segments[idx]["translated"] = segments[idx].get("text", "")

        print(f"[llama-server翻译] 重新翻译完成")
        return segments

    def translate_batch(self, segments, source_lang="en", target_lang="zh", progress_callback=None, trans_params: TransParams = None):
        if trans_params is None:
            trans_params = TransParams()

        total_segments = len(segments)
        print(f"[llama-server翻译] 开始单条翻译，共 {total_segments} 个片段")
        print(f"[llama-server翻译] 最大输出: {trans_params.max_output_tokens} tokens")

        batch_start_time = time.time()

        segments, translated_count, untranslated_indices, context_cache = self._translate_initial(
            segments, source_lang, target_lang, trans_params, progress_callback
        )

        print(f"[llama-server翻译] 单条翻译完成，共翻译 {total_segments} 个片段")

        segments, failed_indices = self._validate_translations(segments, source_lang, target_lang)

        if failed_indices:
            print(f"[llama-server翻译] 发现 {len(failed_indices)} 个片段未翻译或翻译失败，开始重新翻译...")
            segments = self._retry_untranslated(segments, failed_indices, source_lang, target_lang, trans_params, context_cache)
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
    model = LlamaServerManager.find_model_path(None)
    if model:
        print(f"[翻译] 找到本地模型: {model}")
        return os.path.dirname(model)
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
            text = recognized_result.get('text', '')
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
            'text': recognized_result.get('text', ''),
            'start': 0,
            'end': 0
        }]
    
    for seg in segments:
        if 'text' in seg:
            original_text = seg.get('text', '')
            seg['original_text'] = original_text
    
    server_params = server_params or ServerParams.from_dict(config.get_all())
    translator = LlamaCppTranslator(server_params=server_params)

    try:
        translated_segments = translator.translate_batch(
            segments,
            source_lang=source_language,
            target_lang=target_language,
            progress_callback=progress_callback,
            trans_params=trans_params
        )
        
        recognized_result['segments'] = translated_segments
        
        is_cjk_target = target_language and target_language.startswith(('zh', 'ja', 'ko'))
        separator = '' if is_cjk_target else ' '
        translated_text = separator.join([seg.get('translated', seg.get('text', '')) for seg in translated_segments])
        recognized_result['translated_text'] = translated_text
        
        return recognized_result
    finally:
        clear_translator_cache(translator._server_manager)
