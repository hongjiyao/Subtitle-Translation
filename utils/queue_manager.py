#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""队列管理器模块"""
import os
import time
import datetime
import gc

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
from utils.subtitle_generator import generate_subtitle

VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.mpg', '.mpeg']
MAX_FILE_SIZE = 10 * 1024 * 1024 * 1024

def log(msg):
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

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
        return self.video_queue
    
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
            
            # 使用增强版语音识别函数，支持强制对齐
            self.add_log("使用增强版语音识别（强制对齐）...")
            recognized = recognize_speech_enhanced(
                audio_path, params['model'],
                detected_language=None if src_lang == 'auto' else src_lang,
                device_choice=params['device'],
                progress_callback=None,
                word_timestamps=params.get('word_timestamps', True),
                whisperx_batch_size=params.get('speech_batch_size', 16),
                vad_threshold=params.get('whisperx_vad_onset', 0.3),
                enable_alignment=params.get('enable_forced_alignment', True),
                enable_segmentation=True,
                segmentation_options=config.get_segmentation_options()
            )
            
            self.add_log(f"语音识别完成，语言: {recognized['language']}")
            if use_enhanced:
                seg_count = len(recognized.get('segments', []))
                self.add_log(f"生成 {seg_count} 个对齐段落")
            
            # 保存原始语音识别结果到输出文件夹
            try:
                base_name = os.path.splitext(os.path.basename(video_file))[0]
                raw_recognition_path = os.path.join(OUTPUT_DIR, f"{base_name}_raw_recognition.json")
                import json
                with open(raw_recognition_path, 'w', encoding='utf-8') as f:
                    json.dump(recognized, f, ensure_ascii=False, indent=2)
                self.add_log(f"原始语音识别结果已保存: {raw_recognition_path}")

                # 同时保存原文字幕文件
                original_subtitle_path = os.path.join(OUTPUT_DIR, f"{base_name}_original_subtitles.srt")
                generate_subtitle(recognized, original_subtitle_path)
                self.add_log(f"原文字幕已保存: {original_subtitle_path}")
            except Exception as e:
                self.add_log(f"保存原始语音识别结果失败: {str(e)}")
            
            clear_model_cache()
            self._cleanup(params['device'])
            
            # 3. 翻译
            translated = recognized
            if recognized['language'] != tgt_lang:
                log("[阶段] 3. 翻译")
                self.add_log("3. 翻译...")
                try:
                    translated = translate_text(
                        recognized, translator, device_choice=params['device'],
                        target_language=tgt_lang,
                        batch_size=params.get('translation_batch_size', 4096),
                        context_size=params.get('translation_context_size', 4096)
                    )
                    self.add_log("翻译完成")
                except Exception as e:
                    if "out of memory" in str(e).lower():
                        self.add_log("[内存] 不足，减小token限制重试")
                        translated = translate_text(
                            recognized, translator, device_choice=params['device'],
                            target_language=tgt_lang,
                            batch_size=params.get('translation_batch_size', 4096) // 2,
                            context_size=params.get('translation_context_size', 4096)
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
            generate_subtitle(translated, output_path)
            self.add_log(f"字幕生成完成，总耗时: {time.time() - start:.2f}s")
            
            # 清理临时文件
            if os.path.exists(TEMP_DIR):
                for f in os.listdir(TEMP_DIR):
                    try:
                        os.remove(os.path.join(TEMP_DIR, f))
                    except:
                        pass
            self._cleanup(params['device'])
            
            if os.path.exists(output_path):
                return True, "处理成功！", output_path, self.logs
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
