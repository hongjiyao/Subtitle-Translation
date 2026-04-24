# -*- coding: utf-8 -*-


import os
import time
import datetime
import gc
import re

from utils.llama_server_manager import LlamaServerManager
from utils.language_ratio_detector import check_translation_success, get_translation_quality_info

# 强制使用 UTF-8 编码
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['LC_ALL'] = 'en_US.UTF-8'
os.environ['LANG'] = 'en_US.UTF-8'

# 设置HF-Mirror作为下载源
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TRANSFORMERS_OFFLINE"] = "1"  # 强制离线模式，禁止自动下载
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.environ["HUGGINGFACE_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

from config import MODEL_CACHE_DIR, PROJECT_ROOT


def timestamp_print(message):
    """带时间戳的打印函数"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


def clear_translator_cache():
    """清空翻译模型缓存以释放内存并停止服务器进程"""
    # 停止llama-server进程
    try:
        import subprocess
        # 使用taskkill命令强制停止llama-server进程
        subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"], capture_output=True, text=True)
        timestamp_print("[llama-server] 已停止服务器进程")
    except Exception as e:
        timestamp_print(f"[llama-server] 停止服务器时出错: {e}")
    
    gc.collect()
    timestamp_print("[内存管理] 已执行垃圾回收")


class LlamaCppTranslator:
    """使用 llama-server HTTP API 运行 GGUF 模型的翻译器"""

    def __init__(self, model_path=None):
        # 翻译的系统提示词
        system_prompt = "You are a translator. Your only task is to translate the given text from the source language to the target language. Output only the translation, nothing else. Do not include any instructions or explanations in your response."
        # 使用默认端口
        self.server_manager = LlamaServerManager(system_prompt=system_prompt, port=8080)
        
        from config import config
        self.context_size = config.get('translation_context_size', 8192)
        
        timestamp_print(f"[llama-server翻译] 使用 HTTP API: {self.server_manager.host}:{self.server_manager.port}")
        timestamp_print(f"[llama-server翻译] 模型: {self.server_manager.model_path}")
        timestamp_print(f"[llama-server翻译] 默认上下文大小: {self.context_size}")

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
        """预处理文本，压缩重复字符序列
        对于包含重复字符的文本（如"ああああ..."、"うんうんうん..."），进行压缩处理
        将重复序列缩减为仅保留指定数量（默认1个）
        避免占用过多 token 导致其他文本无法翻译
        """
        if not text:
            return text
        # 移除所有空格（包括普通空格、制表符等）
        text = text.replace(' ', '')
        text = text.replace('\t', '')
        text = text.replace('\n', '')
        text = text.replace('\r', '')
        # 移除所有标点符号
        # 定义需要移除的标点符号列表（包含中英文、日语等所有语言的标点）
        punctuation = "。，！？；：\"'（）《》【】…——.,!?:;\"'()<>[]...--「」、！？；：""''（）《》【】…——～～～``''“”‘’〝〞〔〕々〃☆★○●◎◇◆□■△▲▽▼※〓♠♥♦♣♂♀♪♫€¥£¢‰№↑↓←→↘↙Ψ※㊣々♀♂∞①ㄨ≡╬"  
        # 逐个移除标点符号
        for char in punctuation:
            text = text.replace(char, '')
        return self.compress_repeated_sequences(text, keep_count)

    def translate(self, text, source_lang="en", target_lang="zh", context_size=20000):
        """翻译单个文本"""
        # 直接调用批量翻译的回退方法
        return self._translate_multi_fallback([text], source_lang, target_lang, context_size)[0]

    def _translate_multi_fallback(self, texts, source_lang="en", target_lang="zh", context_size=20000, contexts=None):
        """使用单次调用模式进行翻译"""
        # 预处理文本，截断超长重复内容
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # 构建翻译提示
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

        if len(processed_texts) == 1:
            # 单条翻译提示词 - 明确要求只输出翻译
            text_length = len(processed_texts[0])
            if contexts and contexts[0]:
                if text_length <= 5:
                    # 短文本的特殊提示词
                    prompt_lines = [f"<|startoftext|>Translate to {target_lang_name}, considering the context:",
                                   f"Context: {contexts[0]}",
                                   f"Text to translate: {processed_texts[0]}",
                                   "Output only the translation, nothing else.",
                                   f"IMPORTANT: Even if the text is very short (like interjections, onomatopoeia, or single words),",
                                   f"provide a proper translation in {target_lang_name}.",
                                   "Do not leave it untranslated or output the original text.",
                                   "<|extra_0|>"]
                else:
                    # 长文本的常规提示词
                    prompt_lines = [f"<|startoftext|>Translate to {target_lang_name}, considering the context:",
                                   f"Context: {contexts[0]}",
                                   f"Text to translate: {processed_texts[0]}",
                                   "Output only the translation, nothing else.",
                                   f"For short texts and sound words, provide appropriate translations in {target_lang_name}.",
                                   "<|extra_0|>"]
            else:
                if text_length <= 5:
                    # 短文本的特殊提示词
                    prompt_lines = [f"<|startoftext|>Translate to {target_lang_name}: {processed_texts[0]}",
                                   "Output only the translation, nothing else.",
                                   f"IMPORTANT: Even if the text is very short (like interjections, onomatopoeia, or single words),",
                                   f"provide a proper translation in {target_lang_name}.",
                                   "Do not leave it untranslated or output the original text.",
                                   "<|extra_0|>"]
                else:
                    # 长文本的常规提示词
                    prompt_lines = [f"<|startoftext|>Translate to {target_lang_name}: {processed_texts[0]}",
                                   "Output only the translation, nothing else.",
                                   f"For short texts and sound words, provide appropriate translations in {target_lang_name}.",
                                   "<|extra_0|>"]
            prompt = "\n".join(prompt_lines)
        else:
            # 批量翻译提示词 - 明确要求只输出翻译列表
            num_texts = len(processed_texts)
            prompt_lines = [f"<|startoftext|>Translate the following {num_texts} lines to {target_lang_name}.",
                           "Consider the context for each line if provided.",
                           "Output ONLY the translations in numbered list format, nothing else.",
                           "Do not include any explanations."]
            
            for i, (text, context) in enumerate(zip(processed_texts, contexts or [None]*len(texts)), 1):
                if context:
                    prompt_lines.append(f"{i}. Context: {context}")
                    prompt_lines.append(f"   Text: {text}")
                else:
                    prompt_lines.append(f"{i}. {text}")
            
            prompt_lines.append("<|extra_0|>")
            prompt = "\n".join(prompt_lines)

        # 打印输入模型的文本
        timestamp_print("[模型输入] 发送给翻译模型的完整输入:")
        print(prompt)
        print("[模型输入结束]")

        try:
            from config import config
            temperature = config.get('translation_temperature', 0.05)
            top_k = config.get('translation_top_k', 40)
            top_p = config.get('translation_top_p', 0.95)
            repetition_penalty = config.get('translation_repetition_penalty', 1.0)
            
            self.server_manager.ensure_server_running()
            output = self.server_manager.send_request(
                prompt,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repeat_penalty=repetition_penalty,
                n_predict=-1,
                timeout=300
            )
            
            if output is None:
                raise RuntimeError("llama-server 请求返回为空")

            output = output.strip()
            
            # 打印模型输出的原始文本
            timestamp_print("[模型输出] 翻译模型的原始输出:")
            print(output)
            print("[模型输出结束]")
            
            special_tokens = [
                "<|startoftext|>", "<|endoftext|>", "<|eos|>", 
                "<|extra_0|>", "<|extra_4|>", "<|pad|>",
                " [end of text]", "[end of text]"
            ]
            for token in special_tokens:
                output = output.replace(token, "")
            
            if len(texts) == 1:
                translation = output.strip()
                # 过滤掉提示词相关内容
                skip_patterns = [
                    "instructions", "使用说明", "完整翻译",
                    "保留所有", "提供完整", "结构不变",
                    "punctuation", "标点符号", "special characters",
                    "even if it", "no matter how short",
                    "context:", "上下文:", "previous:", "next:", "上文:", "下文:"
                ]
                lower_trans = translation.lower()
                for skip in skip_patterns:
                    if skip.lower() in lower_trans and len(translation) > 100:
                        timestamp_print(f"[翻译诊断] 输出包含提示词 '{skip}'，返回原文")
                        translation = texts[0]
                        break
                # 提取纯翻译内容，去除上下文标记
                # 查找 "Text: " 或 "文本：" 标记
                text_markers = ["Text: ", "text: ", "文本：", "文本:"]
                for marker in text_markers:
                    if marker in translation:
                        translation = translation.split(marker)[-1].strip()
                        break
                translation = " ".join(translation.split())
                return [translation]
            else:
                # 解析带编号的翻译结果
                lines = [line.strip() for line in output.split('\n') if line.strip()]
                translations = []

                # 首先尝试按编号解析 (如 "1. 翻译内容" 或 "1) 翻译内容")
                numbered_trans = {}
                # 同时收集所有非编号行，用于回退解析
                non_numbered_lines = []

                for line in lines:
                    # 匹配编号格式：数字 + 标点 + 空格（编号必须是独立的）
                    # 支持: 1. 内容, 1) 内容, 1: 内容, 1] 内容, 15"内容
                    match = re.match(r'^(\d+)[.\)\:\]]\s*["\']?\s*(.+)$', line)
                    if match:
                        id_num = int(match.group(1))
                        content = match.group(2).strip()
                        # 移除末尾可能残留的引号
                        content = re.sub(r'["\']+$', '', content).strip()
                        
                        # 提取纯翻译内容，去除上下文标记
                        # 1. 首先查找 "Text: " 或 "文本：" 标记
                        text_markers = ["Text: ", "text: ", "文本：", "文本:"]
                        text_found = False
                        for marker in text_markers:
                            if marker in content:
                                content = content.split(marker)[-1].strip()
                                text_found = True
                                break
                        
                        # 2. 如果没有找到文本标记，尝试直接提取翻译内容
                        if not text_found:
                            # 去除上下文相关标记
                            context_markers = ["Context:", "context:", "Previous:", "previous:", "Next:", "next:", "上文:", "下文:"]
                            # 清理所有上下文标记
                            for marker in context_markers:
                                # 移除包含标记的部分
                                parts = content.split(marker)
                                # 只保留不包含标记的部分
                                clean_parts = []
                                for part in parts:
                                    # 检查该部分是否包含其他上下文标记
                                    has_marker = False
                                    for m in context_markers:
                                        if m in part:
                                            has_marker = True
                                            break
                                    if not has_marker:
                                        clean_parts.append(part)
                                content = " ".join(clean_parts).strip()
                        
                        # 3. 进一步清理，去除可能的标记残留
                        # 处理多行格式，提取纯翻译内容
                        lines = content.split('\n')
                        clean_lines = []
                        for line in lines:
                            line = line.strip()
                            if not line:
                                continue
                            # 跳过上下文标记行
                            if any(marker in line for marker in ["Context:", "context:", "Previous:", "previous:", "Next:", "next:", "上文:", "下文:"]):
                                continue
                            # 提取文本标记后的内容
                            if "文本：" in line:
                                line = line.split("文本：")[-1].strip()
                            if "Text:" in line:
                                line = line.split("Text:")[-1].strip()
                            # 清理可能的残留标记
                            line = re.sub(r'^(Context|context|Previous|previous|Next|next|上文|下文|Text|text|文本):\s*', '', line)
                            line = re.sub(r'\s*(Context|context|Previous|previous|Next|next|上文|下文|Text|text|文本):.*$', '', line)
                            if line:
                                clean_lines.append(line)
                        content = " ".join(clean_lines)
                        # 4. 清理多余的空白字符
                        content = " ".join(content.split())
                        
                        # 4. 清理多余的空白字符
                        content = " ".join(content.split())
                        
                        if content:
                            numbered_trans[id_num] = content
                    else:
                        # 收集非编号行
                        lower_line = line.lower()
                        # 过滤掉提示词相关的内容（中英文）
                        skip_patterns = [
                            "translate", "translations", "input",
                            "instructions", "使用说明", "完整翻译",
                            "保留所有", "提供完整", "结构不变",
                            "punctuation", "标点符号", "special characters"
                        ]
                        if not any(skip in lower_line for skip in skip_patterns):
                            cleaned = line.strip()
                            if cleaned:
                                non_numbered_lines.append(cleaned)

                # 按顺序提取翻译
                for i in range(1, len(texts) + 1):
                    if i in numbered_trans:
                        translations.append(numbered_trans[i])
                    else:
                        translations.append(None)

                # 如果编号解析缺失，尝试使用非编号行填充
                if None in translations and non_numbered_lines:
                    timestamp_print(f"[llama-server翻译诊断] 尝试使用非编号行填充缺失的翻译")
                    line_idx = 0
                    for i in range(len(translations)):
                        if translations[i] is None and line_idx < len(non_numbered_lines):
                            translations[i] = non_numbered_lines[line_idx]
                            line_idx += 1

                # 诊断信息：显示解析结果
                timestamp_print(f"[llama-server翻译诊断] 输入片段数: {len(texts)}, 解析到翻译数: {len([t for t in translations if t is not None])}")
                timestamp_print(f"[llama-server翻译诊断] 模型输出行数: {len(lines)}")

                # 如果编号解析缺失太多，尝试直接按行解析
                missing_count = translations.count(None)
                if missing_count > len(texts) * 0.3:  # 缺失超过30%
                    timestamp_print(f"[llama-server翻译] 编号解析缺失 {missing_count} 个，尝试直接按行解析")
                    timestamp_print(f"[llama-server翻译诊断] 前5行输出: {lines[:5]}")
                    translations = []
                    for line in lines:
                        lower_line = line.lower()
                        # 跳过提示词（中英文）
                        skip_patterns = [
                            "translate", "translations", "input", "instructions",
                            "使用说明", "完整翻译", "保留所有", "提供完整", "结构不变"
                        ]
                        if any(skip in lower_line for skip in skip_patterns):
                            continue
                        # 只移除独立的编号前缀（数字 + 标点 + 空格）
                        cleaned = re.sub(r'^\d+[\.\)\:\]]\s*', '', line).strip()
                        if cleaned:
                            translations.append(" ".join(cleaned.split()))
                    timestamp_print(f"[llama-server翻译诊断] 按行解析后得到 {len(translations)} 个翻译")
                
                # 填充缺失的翻译，并记录哪些使用了原文
                filled_count = 0
                for i in range(len(translations)):
                    if translations[i] is None:
                        timestamp_print(f"[llama-server翻译诊断] 片段 {i+1} 使用原文填充: {texts[i][:30]}...")
                        translations[i] = texts[i]
                        filled_count += 1
                
                if filled_count > 0:
                    timestamp_print(f"[llama-server翻译诊断] 共 {filled_count} 个片段使用原文填充")
                
                # 确保数量正确
                while len(translations) < len(texts):
                    idx = len(translations)
                    timestamp_print(f"[llama-server翻译诊断] 片段 {idx+1} 缺失，使用原文: {texts[idx][:30]}...")
                    translations.append(texts[idx])
                
                if len(translations) > len(texts):
                    translations = translations[:len(texts)]
                
                return translations

        except Exception as e:
            raise RuntimeError(f"llama-server 翻译失败: {str(e)}")







    def translate_batch(self, segments, source_lang="en", target_lang="zh", progress_callback=None, batch_size=None, context_size=2048, max_output_tokens=8000):
        from config import config
        
        # 相似度计算缓存
        similarity_cache = {}
        
        def calc_similarity(text1, text2):
            # 生成缓存键
            cache_key = (text1, text2)
            if cache_key in similarity_cache:
                return similarity_cache[cache_key]
                
            if not text1 or not text2:
                result = 0.0
                similarity_cache[cache_key] = result
                return result
            
            # 早期退出：长度差异过大
            len1 = len(text1)
            len2 = len(text2)
            if max(len1, len2) > min(len1, len2) * 3:
                # 长度差异超过3倍，相似度肯定很低
                result = 0.1
                similarity_cache[cache_key] = result
                return result
            
            # 早期退出：完全相同
            if text1 == text2:
                result = 1.0
                similarity_cache[cache_key] = result
                return result
            
            def detect_language(text):
                """简单的语言检测
                
                Args:
                    text: 待检测的文本
                    
                Returns:
                    语言代码 ('ja' for Japanese, 'zh' for Chinese, 'en' for English, 'other' for others)
                """
                import re
                # 检测中文字符
                if re.search(r'[\u4e00-\u9fff]', text):
                    return 'zh'
                # 检测日文字符
                elif re.search(r'[\u3040-\u30ff]', text):
                    return 'ja'
                # 检测英文字符
                elif re.search(r'[a-zA-Z]', text):
                    return 'en'
                else:
                    return 'other'
            
            def normalize(text, lang=None):
                """增强的归一化函数，支持多语言
                
                Args:
                    text: 待归一化的文本
                    lang: 语言代码
                    
                Returns:
                    归一化后的文本
                """
                import re
                # 转换为小写
                text = text.lower()
                # 移除多余空格
                text = re.sub(r'\s+', ' ', text).strip()
                
                # 根据语言进行特殊处理
                if lang == 'ja':
                    # 保留日语假名和汉字
                    text = re.sub(r'[^\w\u3040-\u30ff\u4e00-\u9fff]', '', text)
                elif lang == 'zh':
                    # 保留汉字和基本标点
                    text = re.sub(r'[^\w\u4e00-\u9fff]', '', text)
                elif lang == 'en':
                    # 保留英文单词
                    text = re.sub(r'[^\w]', '', text)
                else:
                    # 通用处理
                    text = re.sub(r'[^\w\u4e00-\u9fff\u3040-\u30ff]', '', text)
                
                return text
            
            def levenshtein_distance(s1, s2):
                """计算Levenshtein距离
                
                Args:
                    s1: 字符串1
                    s2: 字符串2
                    
                Returns:
                    Levenshtein距离
                """
                if len(s1) < len(s2):
                    return levenshtein_distance(s2, s1)
                if len(s2) == 0:
                    return len(s1)
                previous_row = range(len(s2) + 1)
                for i, c1 in enumerate(s1):
                    current_row = [i + 1]
                    for j, c2 in enumerate(s2):
                        insertions = previous_row[j + 1] + 1
                        deletions = current_row[j] + 1
                        substitutions = previous_row[j] + (c1 != c2)
                        current_row.append(min(insertions, deletions, substitutions))
                    previous_row = current_row
                return previous_row[-1]
            
            def calculate_levenshtein_similarity(s1, s2):
                """计算基于Levenshtein距离的相似度
                
                Args:
                    s1: 字符串1
                    s2: 字符串2
                    
                Returns:
                    相似度分数 (0.0-1.0)
                """
                if not s1 or not s2:
                    return 0.0
                distance = levenshtein_distance(s1, s2)
                max_length = max(len(s1), len(s2))
                if max_length == 0:
                    return 1.0
                return 1.0 - (distance / max_length)
            
            def calculate_word_similarity(text1, text2, lang1, lang2):
                """计算词级相似度
                
                Args:
                    text1: 文本1
                    text2: 文本2
                    lang1: 文本1的语言
                    lang2: 文本2的语言
                    
                Returns:
                    词级相似度分数 (0.0-1.0)
                """
                import re
                # 简单的词分割
                if lang1 == 'zh' or lang2 == 'zh':
                    # 中文按字符分割
                    words1 = list(text1)
                    words2 = list(text2)
                else:
                    # 其他语言按空格分割
                    words1 = re.findall(r'\w+', text1)
                    words2 = re.findall(r'\w+', text2)
                
                if not words1 or not words2:
                    return 0.0
                
                # 计算共同词的比例
                common_words = set(words1) & set(words2)
                if not common_words:
                    return 0.0
                
                return len(common_words) / max(len(words1), len(words2))
            
            def calculate_length_factor(text1, text2):
                """计算长度因子，考虑文本长度差异
                
                Args:
                    text1: 文本1
                    text2: 文本2
                    
                Returns:
                    长度因子 (0.0-1.0)
                """
                len1 = len(text1)
                len2 = len(text2)
                if len1 == 0 or len2 == 0:
                    return 0.0
                
                # 计算长度比例
                ratio = min(len1, len2) / max(len1, len2)
                # 对于短文本，长度差异的影响更大
                if max(len1, len2) < 10:
                    return ratio * 0.8 + 0.2
                else:
                    return ratio * 0.5 + 0.5
            
            # 检测语言
            lang1 = detect_language(text1)
            lang2 = detect_language(text2)
            
            # 归一化文本
            norm1 = normalize(text1, lang1)
            norm2 = normalize(text2, lang2)
            
            if not norm1 or not norm2:
                return 0.0
            
            # 计算SequenceMatcher相似度
            from difflib import SequenceMatcher
            sequence_similarity = SequenceMatcher(None, norm1, norm2).ratio()
            
            # 计算Levenshtein相似度
            levenshtein_sim = calculate_levenshtein_similarity(norm1, norm2)
            
            # 计算词级相似度
            word_similarity = calculate_word_similarity(norm1, norm2, lang1, lang2)
            
            # 计算长度因子
            length_factor = calculate_length_factor(norm1, norm2)
            
            # 加权计算总相似度
            # 对于短文本，词级相似度权重更高
            if max(len(norm1), len(norm2)) < 10:
                # 短文本的特殊权重
                if max(len(norm1), len(norm2)) <= 5:
                    # 极短文本（1-5个字符）
                    total_similarity = (
                        sequence_similarity * 0.2 +
                        levenshtein_sim * 0.4 +
                        word_similarity * 0.3 +
                        length_factor * 0.1
                    )
                else:
                    # 中等长度文本（6-9个字符）
                    total_similarity = (
                        sequence_similarity * 0.3 +
                        levenshtein_sim * 0.3 +
                        word_similarity * 0.3 +
                        length_factor * 0.1
                    )
            else:
                # 长文本的常规权重
                total_similarity = (
                    sequence_similarity * 0.4 +
                    levenshtein_sim * 0.3 +
                    word_similarity * 0.2 +
                    length_factor * 0.1
                )
            
            # 缓存结果
            cache_key = (text1, text2)
            similarity_cache[cache_key] = total_similarity
            
            return total_similarity
        
        def extract_context(segments, index, window_size=2):
            """提取上下文信息
            
            Args:
                segments: 所有片段列表
                index: 当前片段的索引
                window_size: 前后上下文的窗口大小
                
            Returns:
                上下文文本
            """
            # 对于短文本，不使用上下文，直接翻译
            current_text = segments[index].get('text', '')
            if len(current_text) <= 5:
                return ""
                
            context = []
            # 提取前上下文
            for i in range(max(0, index - window_size), index):
                context.append(f"Previous: {segments[i]['text']}")
            # 提取后上下文
            for i in range(index + 1, min(len(segments), index + window_size + 1)):
                context.append(f"Next: {segments[i]['text']}")
            return " ".join(context)
        
        total_segments = len(segments)
        timestamp_print(f"[llama-server翻译] 开始单条翻译，共 {total_segments} 个片段")
        timestamp_print(f"[llama-server翻译] 上下文大小: {context_size}, 最大输出: {max_output_tokens} tokens")
        
        # 检查是否需要重置会话
        from config import config
        reset_session = config.get('translation_reset_session', True)
        if reset_session:
            timestamp_print("[llama-server翻译] 重置会话状态，确保全新的翻译环境...")
            self.server_manager.reset_session()
        else:
            # 确保服务器运行
            self.server_manager.ensure_server_running()
        
        processed_count = 0
        for i, segment in enumerate(segments):
            text = segment["text"]
            # 提取上下文信息
            context = extract_context(segments, i)
            
            timestamp_print(f"[llama-server翻译] 处理片段 {i+1}/{total_segments}: {text[:30]}...")
            
            max_retries = 3
            retry_count = 0
            translation = None
            
            while retry_count < max_retries:
                try:
                    # 单条翻译，使用上下文
                    translation = self._translate_multi_fallback([text], source_lang, target_lang, context_size, [context])[0]
                    
                    # 检查翻译结果
                    similarity = calc_similarity(text, translation)
                    # 计算文本长度
                    text_length = len(text)
                    
                    # 自适应阈值计算
                    def calculate_adaptive_threshold(text, source_lang, target_lang):
                        """计算自适应阈值
                        
                        Args:
                            text: 待翻译文本
                            source_lang: 源语言
                            target_lang: 目标语言
                            
                        Returns:
                            自适应阈值
                        """
                        import re
                        text_length = len(text)
                        
                        # 基础阈值 - 基于文本长度
                        if text_length <= 2:
                            base_threshold = 0.9  # 极短文本，允许较高相似度
                        elif text_length <= 5:
                            base_threshold = 0.8  # 短文本，阈值较低
                        else:
                            base_threshold = 0.8
                        
                        # 语言调整因子
                        lang_factor = 1.0
                        if source_lang == 'ja' and target_lang == 'zh':
                            # 日译中，相似度可能较高
                            lang_factor = 0.9
                        elif source_lang == 'en' and target_lang == 'zh':
                            # 英译中，相似度可能较低
                            lang_factor = 1.1
                        
                        # 内容类型调整
                        content_factor = 1.0
                        # 检测拟声词
                        if re.search(r'[\u3040-\u30ff]+[！!]+', text):
                            content_factor = 0.7  # 拟声词翻译可能与原文相似
                        # 检测感叹词
                        elif re.search(r'[！!]+$', text):
                            content_factor = 0.8
                        # 检测数字和代码
                        elif re.search(r'^[0-9a-zA-Z]+$', text):
                            content_factor = 1.05  # 数字和代码可能需要完全相同
                        
                        # 计算最终阈值
                        final_threshold = base_threshold * lang_factor * content_factor
                        # 确保阈值在合理范围内
                        final_threshold = max(0.5, min(1.01, final_threshold))
                        return final_threshold
                    
                    threshold = calculate_adaptive_threshold(text, source_lang, target_lang)
                    timestamp_print(f"[llama-server翻译] 片段 {i+1} 相似度: {similarity}, 阈值: {threshold}, 原文: {text[:20]}..., 译文: {translation[:20]}...")
                    
                    if similarity <= threshold:
                        # 翻译成功
                        segment["translated"] = translation
                        timestamp_print(f"[llama-server翻译] 翻译成功: {translation[:30]}...")
                    else:
                        # 翻译失败，使用原文
                        segment["translated"] = text
                        timestamp_print(f"[llama-server翻译] 翻译失败，使用原文")
                    break
                except Exception as e:
                    retry_count += 1
                    timestamp_print(f"[llama-server翻译] 翻译失败，第{retry_count}次重试: {str(e)}")
                    if retry_count >= max_retries:
                        # 多次重试失败，使用原文
                        segment["translated"] = text
                        timestamp_print(f"[llama-server翻译] 多次重试失败，使用原文")
            
            processed_count += 1
            if progress_callback and total_segments > 0:
                progress_callback(int(processed_count / total_segments * 100))
            
            # 打印翻译结果
            original = segment["text"]
            translated = segment.get("translated") or original
            orig_display = original[:50] + "..." if len(original) > 50 else original
            trans_display = translated[:50] + "..." if len(translated) > 50 else translated
            print(f"  [{i+1}/{total_segments}] {orig_display} -> {trans_display}")
        
        timestamp_print(f"[llama-server翻译] 单条翻译完成，共翻译 {total_segments} 个片段")
        print(f"[llama-server翻译] 单条翻译完成，共翻译 {total_segments} 个片段")
        
        # 验证翻译结果
        print(f"[llama-server翻译] 开始验证翻译结果...")
        timestamp_print(f"[llama-server翻译] 开始验证翻译结果...")
        
        # 统计翻译成功和失败的数量
        success_count = 0
        fail_count = 0
        untranslated_indices = []
        
        for i, segment in enumerate(segments):
            original_text = segment["text"]
            translated_text = segment.get("translated", "")
            
            # 使用语言占比检测判断翻译是否成功
            success, target_ratio, lang_counts = check_translation_success(
                original_text, translated_text, source_lang, target_lang, threshold=0.5
            )
            
            if success:
                success_count += 1
            else:
                untranslated_indices.append(i)
                fail_count += 1
                quality_info = get_translation_quality_info(original_text, translated_text, source_lang, target_lang)
                print(f"[llama-server翻译] 片段 {i+1} 翻译失败: {quality_info['message']}, 原文={original_text[:30]}...")
        
        print(f"[llama-server翻译] 验证完成: 成功 {success_count} 个, 失败 {fail_count} 个")
        timestamp_print(f"[llama-server翻译] 验证完成: 成功 {success_count} 个, 失败 {fail_count} 个")
        
        if untranslated_indices:
            print(f"[llama-server翻译] 发现 {len(untranslated_indices)} 个片段未翻译或翻译失败，开始重新翻译...")
            timestamp_print(f"[llama-server翻译] 发现 {len(untranslated_indices)} 个片段未翻译或翻译失败，开始重新翻译...")
            
            # 重新翻译直到成功或达到最大重试次数
            max_total_retries = 10  # 每个片段的最大总重试次数
            current_retries = {idx: 0 for idx in untranslated_indices}
            remaining_indices = untranslated_indices.copy()
            
            while remaining_indices and max(current_retries.values()) < max_total_retries:
                current_batch = remaining_indices.copy()
                remaining_indices = []
                
                for idx in current_batch:
                    segment = segments[idx]
                    text = segment["text"]
                    # 对于短文本，不使用上下文，直接翻译
                    if len(text) <= 5:
                        context = ""
                    else:
                        context = extract_context(segments, idx)
                    
                    current_retries[idx] += 1
                    retry_count = current_retries[idx]
                    
                    if retry_count > max_total_retries:
                        # 达到最大重试次数，使用原文
                        segment["translated"] = text
                        print(f"[llama-server翻译] 片段 {idx+1} 达到最大重试次数 {max_total_retries}，使用原文")
                        timestamp_print(f"[llama-server翻译] 片段 {idx+1} 达到最大重试次数 {max_total_retries}，使用原文")
                        continue
                    
                    print(f"[llama-server翻译] 重新翻译片段 {idx+1}/{total_segments} (第{retry_count}次): {text[:30]}...")
                    timestamp_print(f"[llama-server翻译] 重新翻译片段 {idx+1}/{total_segments} (第{retry_count}次): {text[:30]}...")
                    
                    try:
                        # 单条翻译，使用上下文
                        translation = self._translate_multi_fallback([text], source_lang, target_lang, context_size, [context])[0]
                        
                        # 检查翻译结果
                        similarity = calc_similarity(text, translation)
                        
                        # 自适应阈值计算
                        def calculate_adaptive_threshold(text, source_lang, target_lang):
                            """计算自适应阈值
                            
                            Args:
                                text: 待翻译文本
                                source_lang: 源语言
                                target_lang: 目标语言
                                
                            Returns:
                                自适应阈值
                            """
                            import re
                            text_length = len(text)
                            
                            # 基础阈值 - 基于文本长度
                            if text_length <= 2:
                                base_threshold = 0.9  # 极短文本，允许较高相似度
                            elif text_length <= 5:
                                base_threshold = 0.8  # 短文本，阈值较低
                            else:
                                base_threshold = 0.8
                            
                            # 语言调整因子
                            lang_factor = 1.0
                            if source_lang == 'ja' and target_lang == 'zh':
                                # 日译中，相似度可能较高
                                lang_factor = 0.9
                            elif source_lang == 'en' and target_lang == 'zh':
                                # 英译中，相似度可能较低
                                lang_factor = 1.1
                            
                            # 内容类型调整
                            content_factor = 1.0
                            # 检测拟声词
                            if re.search(r'[\u3040-\u30ff]+[！!]+', text):
                                content_factor = 0.7  # 拟声词翻译可能与原文相似
                            # 检测感叹词
                            elif re.search(r'[！!]+$', text):
                                content_factor = 0.8
                            # 检测数字和代码
                            elif re.search(r'^[0-9a-zA-Z]+$', text):
                                content_factor = 1.05  # 数字和代码可能需要完全相同
                            
                            # 计算最终阈值
                            final_threshold = base_threshold * lang_factor * content_factor
                            # 确保阈值在合理范围内
                            final_threshold = max(0.5, min(1.01, final_threshold))
                            return final_threshold
                        
                        threshold = calculate_adaptive_threshold(text, source_lang, target_lang)
                        print(f"[llama-server翻译] 片段 {idx+1} 相似度: {similarity}, 阈值: {threshold}, 原文: {text[:20]}..., 译文: {translation[:20]}...")
                        timestamp_print(f"[llama-server翻译] 片段 {idx+1} 相似度: {similarity}, 阈值: {threshold}, 原文: {text[:20]}..., 译文: {translation[:20]}...")
                        
                        # 再次验证翻译结果，使用语言占比检测
                        success, target_ratio, lang_counts = check_translation_success(
                            text, translation, source_lang, target_lang, threshold
                        )
                        quality_info = get_translation_quality_info(text, translation, source_lang, target_lang)
                        print(f"[llama-server翻译] 片段 {idx+1} 翻译质量: {quality_info['message']}, 占比: {target_ratio:.2%}, 原文: {text[:20]}..., 译文: {translation[:20]}...")
                        timestamp_print(f"[llama-server翻译] 片段 {idx+1} 翻译质量: {quality_info['message']}, 占比: {target_ratio:.2%}, 原文: {text[:20]}..., 译文: {translation[:20]}...")
                        
                        if success:
                            # 翻译成功
                            segment["translated"] = translation
                            print(f"[llama-server翻译] 重新翻译成功: {translation[:30]}...")
                            timestamp_print(f"[llama-server翻译] 重新翻译成功: {translation[:30]}...")
                        else:
                            # 翻译失败，继续重试
                            print(f"[llama-server翻译] 重新翻译仍失败，将再次尝试")
                            timestamp_print(f"[llama-server翻译] 重新翻译仍失败，将再次尝试")
                            remaining_indices.append(idx)
                    except Exception as e:
                        print(f"[llama-server翻译] 重新翻译失败，第{retry_count}次重试: {str(e)}")
                        timestamp_print(f"[llama-server翻译] 重新翻译失败，第{retry_count}次重试: {str(e)}")
                        # 异常情况下继续重试
                        remaining_indices.append(idx)
                
                # 打印当前批次的重新翻译结果
                for idx in current_batch:
                    original = segments[idx]["text"]
                    translated = segments[idx].get("translated") or original
                    orig_display = original[:50] + "..." if len(original) > 50 else original
                    trans_display = translated[:50] + "..." if len(translated) > 50 else translated
                    print(f"  [{idx+1}/{total_segments}] (重新翻译) {orig_display} -> {trans_display}")
            
            # 处理最终仍失败的片段
            if remaining_indices:
                print(f"[llama-server翻译] 仍有 {len(remaining_indices)} 个片段翻译失败，使用原文")
                timestamp_print(f"[llama-server翻译] 仍有 {len(remaining_indices)} 个片段翻译失败，使用原文")
                for idx in remaining_indices:
                    segments[idx]["translated"] = segments[idx]["text"]
                    text = segments[idx]["text"]
                    print(f"  [{idx+1}/{total_segments}] (最终失败) 使用原文: {text[:50]}...")
            
            print(f"[llama-server翻译] 重新翻译完成")
            timestamp_print(f"[llama-server翻译] 重新翻译完成")
        else:
            print(f"[llama-server翻译] 所有片段翻译成功，无需重新翻译")
            timestamp_print(f"[llama-server翻译] 所有片段翻译成功，无需重新翻译")
        
        return segments


def get_local_model_path(model_path):
    """获取本地模型路径"""
    # 处理各种可能的模型路径格式
    model_name = model_path.replace("tencent/", "").replace("tencent--", "").split("/")[-1]

    # 可能的本地路径列表
    possible_paths = [
        # 直接目录名
        os.path.join(MODEL_CACHE_DIR, model_name),
        # HuggingFace缓存格式
        os.path.join(MODEL_CACHE_DIR, f"tencent--{model_name}"),
        # 子目录格式
        os.path.join(MODEL_CACHE_DIR, "tencent", model_name),
        # 处理 HY-MT1.5-7B-GGUF 的特殊情况
        os.path.join(MODEL_CACHE_DIR, "tencent--HY-MT1.5-7B-GGUF"),
        os.path.join(MODEL_CACHE_DIR, "HY-MT1.5-7B-GGUF"),
    ]

    # 检查所有可能的路径
    for path in possible_paths:
        if os.path.exists(path) and os.path.isdir(path):
            timestamp_print(f"[翻译] 找到本地模型: {path}")
            return path

    # 检查原始路径
    direct_path = os.path.join(MODEL_CACHE_DIR, model_path.replace("/", "--"))
    if os.path.exists(direct_path) and os.path.isdir(direct_path):
        timestamp_print(f"[翻译] 找到本地模型: {direct_path}")
        return direct_path

    return None


def translate_text(recognized_result, model_path, device_choice="auto", progress_callback=None,
                   beam_size=1, max_length=256, target_language="zh", batch_size=2000, context_size=None, max_output_tokens=8000):
    """翻译识别结果"""
    if not target_language:
        target_language = "zh"

    # 获取本地模型路径
    local_model_path = get_local_model_path(model_path)
    if not local_model_path:
        error_msg = f"本地模型不存在: {model_path}，请确保模型已在models目录中"
        timestamp_print(f"[错误信息] {error_msg}")
        raise FileNotFoundError(error_msg)

    # 如果没有指定上下文大小，从配置中获取
    if context_size is None:
        from config import config
        context_size = config.get('translation_context_size', 20000)

    # 使用 llama-server HTTP API 运行 GGUF 模型
    translated_result = translate_with_llama_server(recognized_result, progress_callback, target_language, batch_size, context_size, max_output_tokens)
    
    # 验证翻译结果
    if 'segments' in translated_result:
        has_translation = any('translated' in seg for seg in translated_result['segments'])
        timestamp_print(f"[翻译] 翻译完成，{len(translated_result['segments'])} 个片段，其中 {sum(1 for seg in translated_result['segments'] if 'translated' in seg)} 个片段有翻译")
    
    return translated_result


def translate_with_llama_server(recognized_result, progress_callback, target_language, batch_size=None, context_size=None, max_output_tokens=8000):
    """使用 llama-server HTTP API 运行 GGUF 模型进行翻译"""
    from config import config
    
    # 从配置获取token限制（如果未指定）
    if batch_size is None:
        batch_size = config.get('translation_batch_size', 3500)
    
    if context_size is None:
        context_size = config.get('translation_context_size', 8192)
    
    # 从识别结果中提取源语言
    source_language = recognized_result.get('language', 'auto')
    if source_language == 'auto':
        # 自动检测语言
        if recognized_result.get('text'):
            # 简单语言检测
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
    
    # 提取需要翻译的片段
    segments = recognized_result.get('segments', [])
    if not segments and recognized_result.get('text'):
        # 如果没有段落，创建一个段落
        segments = [{
            'id': 0,
            'text': recognized_result['text'],
            'start': 0,
            'end': 0
        }]
    
    # 过滤文本中的标点符号
    def clean_text_for_translation(text):
        """去除标点符号，只保留文字和空格"""
        import re
        # 去除各种标点符号，保留字母、数字、汉字、假名等
        # 保留日文假名、汉字、韩文、拉丁字母、数字
        cleaned = re.sub(r'[^\u3040-\u309f\u30a0-\u30ff\u4e00-\u9faf\ac00-\ud7af\u0041-\u005a\u0061-\u007a\u0030-\u0039\s]', ' ', text)
        # 去除多余空格
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned
    
    # 对每个片段的文本进行清理
    for seg in segments:
        if 'text' in seg:
            original_text = seg['text']
            seg['original_text'] = original_text  # 保留原文
            seg['text'] = clean_text_for_translation(original_text)
    
    # 初始化翻译器
    translator = LlamaCppTranslator()
    
    try:
        # 执行批量翻译
        translated_segments = translator.translate_batch(
            segments,
            source_lang=source_language,
            target_lang=target_language,
            progress_callback=progress_callback,
            batch_size=batch_size,
            context_size=context_size,
            max_output_tokens=max_output_tokens
        )
        
        # 更新识别结果
        recognized_result['segments'] = translated_segments
        
        # 构建翻译后的完整文本
        translated_text = ''.join([seg.get('translated', seg.get('text', '')) for seg in translated_segments])
        recognized_result['translated_text'] = translated_text
        
        return recognized_result
    finally:
        # 清理资源
        clear_translator_cache()
