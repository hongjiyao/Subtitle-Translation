#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""队列管理器模块"""
import os
import time
import datetime
import gc

# 导入日志模块

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HUGGINGFACE_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.environ["HF_HOME"] = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from config import MODEL_OPTIONS, TEMP_DIR, OUTPUT_DIR, config
from utils.video_processor import extract_audio
from utils.speech_recognizer import recognize_speech, recognize_speech_enhanced, clear_model_cache
from utils.translator import translate_text, clear_translator_cache
from utils.subtitle_generator import generate_subtitle, generate_translated_subtitle, generate_bilingual_subtitle
from utils.punctuation_splitter import split_by_punctuation, ALL_SPLIT_PUNCTUATION

VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.mpg', '.mpeg', '.ts']
MAX_FILE_SIZE = 10 * 1024 * 1024 * 1024

def log(msg):
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

def convert_numpy(obj):
    """将 numpy 类型转换为可序列化的 Python 类型"""
    import numpy as np
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.number):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(item) for item in obj]
    else:
        return obj


class QueueManager:
    def __init__(self):
        self.video_queue = []
        self.processing = False
        self.logs = []
    
    def add_log(self, msg):
        if msg and isinstance(msg, str):
            self.logs.append(msg)
            log(f"[处理] {msg}")
            if len(self.logs) > 500:
                self.logs = self.logs[-500:]
        return "\n".join(self.logs)
    
    def get_queue(self):
        # 返回适合Dataframe的格式：[[id, filename, status, progress], ...]
        return [[i, item['filename'], item['status'], '0%'] for i, item in enumerate(self.video_queue)]
    
    def _validate_file(self, path):
        if not path or not isinstance(path, str):
            return False, "无效路径"
        if not os.path.exists(path):
            return False, f"文件不存在: {path}"
        if os.path.splitext(path)[1].lower() not in VIDEO_EXTENSIONS:
            return False, "不支持的格式"
        if os.path.getsize(path) > MAX_FILE_SIZE:
            return False, "文件过大"
        return True, None
    
    def add_to_queue(self, files, params):
        log(f"\n{'='*80}\n[队列] 添加文件")
        if not files:
            return 0
        
        files = [files] if isinstance(files, str) else files
        processed = []
        for f in files:
            if isinstance(f, str):
                processed.append(f)
            elif hasattr(f, 'name'):
                processed.append(f.name)
            elif isinstance(f, dict):
                processed.append(f.get('path') or f.get('name', ''))
        files = [f for f in processed if f][:50 - len(self.video_queue)]
        
        count = 0
        for f in files:
            valid, err = self._validate_file(f)
            if not valid:
                log(f"[警告] {err}")
                continue
            self.video_queue.append({'file_path': f, 'filename': os.path.basename(f), 'status': '等待中', 'params': params})
            count += 1
            log(f"[成功] 添加: {f}")
        
        log(f"[队列] 已添加 {count} 个文件")
        return count
    
    def remove_from_queue(self, idx):
        log(f"\n{'='*80}\n[队列] 删除索引: {idx}")
        if not self.video_queue:
            return
        try:
            idx = int(idx.strip() if isinstance(idx, str) else idx)
            if 0 <= idx < len(self.video_queue):
                log(f"[成功] 删除: {self.video_queue[idx]['filename']}")
                self.video_queue.pop(idx)
        except Exception as e:
            log(f"[错误] 删除失败: {e}")
    
    def clear_queue(self):
        log(f"\n{'='*80}\n[队列] 清空")
        count = len(self.video_queue)
        self.video_queue = []
        log(f"[成功] 清空 {count} 个文件")
        return count
    
    def _cleanup(self, device):
        gc.collect()
        if device != 'cpu':
            torch.cuda.empty_cache()
        gc.collect()
    
    def process_video(self, video_file, params):
        try:
            if not video_file or not os.path.exists(video_file):
                return False, "文件不存在", None, []
            
            if os.path.splitext(video_file)[1].lower() not in VIDEO_EXTENSIONS:
                return False, "不支持的格式", None, []
            
            # 验证参数
            if params['model'] not in MODEL_OPTIONS:
                return False, "模型选择错误", None, []
            
            translator = params['translator']
            if translator not in ['tencent/HY-MT1.5-7B-GGUF', 'HY-MT1.5-7B-GGUF']:
                return False, f"翻译模型错误: {translator} 不在支持的模型列表中", None, []
            
            output_path = params.get('output_path') or os.path.join(OUTPUT_DIR, f"{os.path.splitext(os.path.basename(video_file))[0]}_subtitles.srt")
            src_lang = params.get('source_language', 'auto')
            tgt_lang = params.get('target_language', 'zh')
            
            self.add_log(f"开始处理: {video_file}")
            os.makedirs(TEMP_DIR, exist_ok=True)
            
            # 1. 提取音频
            log("[阶段] 1. 提取音频")
            self.add_log("1. 提取音频...")
            start = time.time()
            audio_path = extract_audio(video_file)
            self.add_log(f"音频提取完成，耗时: {time.time() - start:.2f}s")
            self._cleanup(params['device'])
            
            # 2. 语音识别（使用增强版）
            log("[阶段] 2. 语音识别")
            self.add_log("2. 语音识别...")
            
            # 检查是否启用增强功能
            use_enhanced = params.get('enable_forced_alignment', True)
            
            # 使用增强版语音识别函数，支持强制对齐和标点还原
            self.add_log("使用增强版语音识别（强制对齐+标点还原）...")
            recognized = recognize_speech_enhanced(
                audio_path, params['model'],
                detected_language=None if src_lang == 'auto' else src_lang,
                device_choice=params['device'],
                progress_callback=None,
                word_timestamps=params.get('word_timestamps', True),
                vad_threshold=params.get('vad_threshold', 0.4),
                min_speech_duration=params.get('vad_min_speech_duration', 1.0),
                max_speech_duration=params.get('vad_max_speech_duration', 30.0),
                min_silence_duration=params.get('vad_min_silence_duration', 1.0),
                speech_pad_ms=params.get('vad_speech_pad_ms', 300),
                prefix_padding_ms=params.get('vad_prefix_padding_ms', 50),
                use_max_poss_sil_at_max_speech=params.get('use_max_poss_sil_at_max_speech', True),
                enable_alignment=params.get('enable_forced_alignment', True),
                enable_whispercd=params.get('enable_whispercd', True),
                enable_punctuate=params.get('enable_punctuate', True),
                neg_threshold=params.get('vad_neg_threshold', None)
            )
            
            self.add_log(f"语音识别完成，语言: {recognized['language']}")
            if use_enhanced:
                seg_count = len(recognized.get('segments', []))
                self.add_log(f"生成 {seg_count} 个对齐段落")
            
            # 保存强制对齐后的语音识别结果到输出文件夹
            try:
                base_name = os.path.splitext(os.path.basename(video_file))[0]
                aligned_recognition_path = os.path.join(OUTPUT_DIR, f"{base_name}_aligned_recognition.json")
                import json
                # 转换 numpy 类型为可序列化的 Python 类型
                recognized_serializable = convert_numpy(recognized)
                # 移除 original_logits 以节省空间（递归处理）
                def remove_logits_recursive(obj):
                    if isinstance(obj, dict):
                        obj.pop('original_logits', None)
                        for v in obj.values():
                            remove_logits_recursive(v)
                    elif isinstance(obj, list):
                        for item in obj:
                            remove_logits_recursive(item)
                    return obj
                recognized_for_save = remove_logits_recursive(convert_numpy(recognized))
                with open(aligned_recognition_path, 'w', encoding='utf-8') as f:
                    json.dump(recognized_for_save, f, ensure_ascii=False, indent=2)
                self.add_log(f"强制对齐后的语音识别结果已保存: {aligned_recognition_path}")
            except Exception as e:
                self.add_log(f"保存强制对齐后的语音识别结果失败: {str(e)}")
            
            clear_model_cache()
            self._cleanup(params['device'])

            # 断句处理：使用标点符号进行断句，并使用字符级别时间戳精确计算时间
            log("[阶段] 2.5 断句处理")
            self.add_log("2.5 断句处理...")

            all_new_segments = []
            for segment in recognized_serializable.get('segments', []):
                text = segment.get("text", "").strip()
                if not text:
                    continue

                # 跳过短于20字符的片段
                if len(text) < 20:
                    all_new_segments.append(segment)
                    continue

                # 获取字符级别的时间戳信息（不包含标点符号）
                chars_info = segment.get("chars", [])

                # 构建字符列表及其时间戳和序号（chars_info中的序号）
                char_list = []
                for idx, char_info in enumerate(chars_info):
                    char_text = char_info.get("char", "")
                    # 排除标点符号
                    if char_text not in ALL_SPLIT_PUNCTUATION:
                        start_time = char_info.get("start", 0.0)
                        end_time = char_info.get("end", 0.0)
                        char_list.append({
                            "char": char_text,
                            "start": start_time,
                            "end": end_time,
                            "idx": idx  # 在chars_info中的序号
                        })

                # 使用标点分割器进行断句
                sentences = split_by_punctuation(text)
                seg_start = segment.get("start", 0.0)
                seg_end = segment.get("end", 0.0)

                for sentence_info in sentences:
                    sentence = sentence_info['text'].strip()
                    # 过滤掉只包含标点符号或非标点字符少于2个的无效句子
                    non_punct_chars = [c for c in sentence if c not in ALL_SPLIT_PUNCTUATION]
                    if len(non_punct_chars) < 2:
                        continue
                    # 获取句子在原文中的字符位置范围
                    sent_start_pos = sentence_info['start']
                    sent_end_pos = sentence_info['end']

                    # 计算句子时间戳
                    # 使用线性比例计算，基于sentence在text中的位置
                    # 不再使用字级时间戳，简化计算逻辑
                    text_length = len(text)
                    seg_duration = seg_end - seg_start

                    # sent_start_pos是sentence在text中的开始位置（字符索引）
                    # sent_end_pos是sentence在text中的结束位置（字符索引）
                    start_ratio = sent_start_pos / text_length if text_length > 0 else 0
                    end_ratio = sent_end_pos / text_length if text_length > 0 else 1

                    # 计算时间戳，使用segment的时间范围作为基础
                    sentence_start = seg_start + seg_duration * start_ratio
                    sentence_end = seg_start + seg_duration * end_ratio

                    all_new_segments.append({
                        "text": sentence,
                        "start": sentence_start,
                        "end": sentence_end
                    })

            # 更新recognized_serializable的segments
            recognized_serializable['segments'] = all_new_segments
            self.add_log(f"断句处理完成，共 {len(all_new_segments)} 个片段")

            # 3. 翻译
            translated = recognized_serializable
            if recognized_serializable['language'] != tgt_lang:
                log("[阶段] 3. 翻译")
                self.add_log("3. 翻译...")
                try:
                    translated = translate_text(
                        recognized_serializable, translator, device_choice=params['device'],
                        target_language=tgt_lang,
                        batch_size=params.get('translation_batch_size', 3500),
                        context_size=params.get('translation_context_size', 8192)
                    )
                    self.add_log("翻译完成")
                except Exception as e:
                    if "out of memory" in str(e).lower():
                        self.add_log("[内存] 不足，减小token限制重试")
                        translated = translate_text(
                            recognized_serializable, translator, device_choice=params['device'],
                            target_language=tgt_lang,
                            batch_size=params.get('translation_batch_size', 3500) // 2,
                            context_size=params.get('translation_context_size', 8192)
                        )
                    else:
                        raise
            else:
                self.add_log(f"跳过翻译，源语言与目标语言相同 ({tgt_lang})")
            
            clear_translator_cache()
            self._cleanup(params['device'])
            
            # 4. 生成字幕
            log("[阶段] 4. 生成字幕")
            self.add_log("4. 生成字幕...")
            
            # 生成原文版本字幕
            original_subtitle_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(os.path.basename(video_file))[0]}_original_subtitles.srt")
            generate_subtitle(
                recognized, 
                original_subtitle_path
            )
            self.add_log(f"原文字幕生成完成: {original_subtitle_path}")
            
            # 生成译文版本字幕
            translated_subtitle_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(os.path.basename(video_file))[0]}_translated_subtitles.srt")
            generate_translated_subtitle(
                translated, 
                translated_subtitle_path
            )
            self.add_log(f"译文字幕生成完成: {translated_subtitle_path}")
            
            # 生成双语版本字幕
            bilingual_subtitle_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(os.path.basename(video_file))[0]}_bilingual_subtitles.srt")
            generate_bilingual_subtitle(
                translated, 
                bilingual_subtitle_path
            )
            self.add_log(f"双语字幕生成完成: {bilingual_subtitle_path}")
            
            self.add_log(f"字幕生成完成，总耗时: {time.time() - start:.2f}s")
            
            # 清理临时文件
            if os.path.exists(TEMP_DIR):
                for f in os.listdir(TEMP_DIR):
                    try:
                        os.remove(os.path.join(TEMP_DIR, f))
                    except:
                        pass
            self._cleanup(params['device'])
            
            # 收集所有生成的字幕文件
            all_outputs = [original_subtitle_path, translated_subtitle_path, bilingual_subtitle_path]
            all_outputs = [f for f in all_outputs if os.path.exists(f)]
            
            if all_outputs:
                return True, "处理成功！", all_outputs, self.logs
            return False, "字幕文件未生成", None, self.logs
            
        except Exception as e:
            import traceback
            log(f"[错误] 处理失败: {e}\n{traceback.format_exc()}")
            self.add_log(f"处理失败: {e}")
            self._cleanup(params.get('device', 'cpu'))
            return False, f"处理错误: {e}", None, self.logs
    
    def process_queue(self):
        log(f"\n{'='*80}\n[队列] 开始处理，共 {len(self.video_queue)} 个文件")
        
        if not self.video_queue:
            yield [], "", "", 0, ""
            return
        
        if self.processing:
            yield [[i['filename'], i['status']] for i in self.video_queue], "", "", 0, ""
            return
        
        self.processing = True
        
        try:
            for i, item in enumerate(self.video_queue):
                self.video_queue[i]['status'] = '处理中'
                log(f"[队列] 处理第 {i+1}/{len(self.video_queue)} 个: {item['filename']}")
                yield [[j['filename'], j['status']] for j in self.video_queue], "", "", 0, ""
                
                self.logs = []
                success, msg, output, logs = self.process_video(item['file_path'], item['params'])
                log(f"[队列] 结果: {msg}")
                
                self.video_queue[i]['status'] = '已完成' if success else '失败'
                yield [[j['filename'], j['status']] for j in self.video_queue], "", "\n".join(logs), 100, ""
                
                gc.collect()
                time.sleep(0.5)
            
            log(f"[队列] 处理完成，共 {len(self.video_queue)} 个文件")
            yield [[i['filename'], i['status']] for i in self.video_queue], "", "\n".join(self.logs), 100, ""
            
        except Exception as e:
            log(f"[错误] 队列处理失败: {e}")
            yield [], "", "", 0, ""
        
        finally:
            self.processing = False
            self.logs = []
            clear_model_cache()
            clear_translator_cache()

queue_manager = QueueManager()
