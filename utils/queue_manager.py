#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""队列管理器模块"""
import os
import time
import gc
import threading



import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from config import MODEL_OPTIONS, TEMP_DIR, OUTPUT_DIR, config, CdParams, TransParams, ServerParams, PARAM_DEFINITIONS
from utils.video_processor import extract_audio
from utils.speech_recognizer import recognize_speech_enhanced, clear_model_cache
from utils.translator import translate_text, clear_translator_cache
from utils.subtitle_generator import generate_subtitle, generate_translated_subtitle, generate_bilingual_subtitle

VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.mpg', '.mpeg', '.ts']
MAX_FILE_SIZE = 10 * 1024 * 1024 * 1024

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
        self.prints = []
        self._lock = threading.Lock()
        self._cancel_flag = False
    
    def cancel_processing(self):
        self._cancel_flag = True
        print("[队列] 收到取消请求")

    def _check_cancelled(self):
        if self._cancel_flag:
            self._cancel_flag = False
            return True
        return False

    def add_print(self, msg):
        if msg and isinstance(msg, str):
            self.prints.append(msg)
            print(f"[处理] {msg}")
            if len(self.prints) > 500:
                self.prints = self.prints[-500:]
        return "\n".join(self.prints)
    
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
    
    def get_queue_statuses(self):
        with self._lock:
            return [f"{item['filename']}: {item['status']}" for item in self.video_queue]
    
    def add_to_queue(self, files, params):
        print(f"\n{'='*80}\n[队列] 添加文件")
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
        with self._lock:
            for f in files:
                valid, err = self._validate_file(f)
                if not valid:
                    print(f"[警告] {err}")
                    continue
                self.video_queue.append({'file_path': f, 'filename': os.path.basename(f), 'status': '等待中', 'params': params})
                count += 1
                print(f"[成功] 添加: {f}")
        
        print(f"[队列] 已添加 {count} 个文件")
        return count
    
    def remove_from_queue(self, idx):
        print(f"\n{'='*80}\n[队列] 删除索引: {idx}")
        if not self.video_queue:
            return
        try:
            idx = int(idx.strip() if isinstance(idx, str) else idx)
            with self._lock:
                if 0 <= idx < len(self.video_queue):
                    print(f"[成功] 删除: {self.video_queue[idx]['filename']}")
                    self.video_queue.pop(idx)
        except Exception as e:
            print(f"[错误] 删除失败: {e}")
    
    def clear_queue(self):
        print(f"\n{'='*80}\n[队列] 清空")
        with self._lock:
            count = len(self.video_queue)
            self.video_queue = []
        print(f"[成功] 清空 {count} 个文件")
        return count
    
    def _cleanup(self, device):
        gc.collect()
        if device != 'cpu':
            torch.cuda.empty_cache()
    
    def process_video(self, video_file, params):
        temp_files = []
        try:
            if not video_file or not os.path.exists(video_file):
                return False, "文件不存在", None, []
            
            if os.path.splitext(video_file)[1].lower() not in VIDEO_EXTENSIONS:
                return False, "不支持的格式", None, []
            
            # 验证参数
            if params['model'] not in MODEL_OPTIONS:
                return False, "模型选择错误", None, []
            
            translator = params['translator']
            if translator not in PARAM_DEFINITIONS['translator']['options']:
                return False, f"翻译模型错误: {translator} 不在支持的模型列表中", None, []
            
            output_path = params.get('output_path') or os.path.join(OUTPUT_DIR, f"{os.path.splitext(os.path.basename(video_file))[0]}_subtitles.srt")
            src_lang = params.get('source_language', 'ja')
            tgt_lang = params.get('target_language', 'zh')
            cd_params = CdParams.from_dict(params)
            trans_params = TransParams.from_dict(params)
            server_params = ServerParams.from_dict(params)
            
            self.add_print(f"开始处理: {video_file}")
            os.makedirs(TEMP_DIR, exist_ok=True)
            
            # 1. 提取音频
            print("[阶段] 1. 提取音频")
            self.add_print("1. 提取音频...")
            start = time.time()
            audio_path = extract_audio(video_file)
            temp_files.append(audio_path)
            self.add_print(f"音频提取完成，耗时: {time.time() - start:.2f}s")

            if self._check_cancelled():
                self.add_print("处理已被用户取消")
                return False, "处理已取消", None, self.prints

            # 2. 语音识别（使用增强版）
            print("[阶段] 2. 语音识别")
            self.add_print("2. 语音识别...")
            
            # 检查是否启用增强功能
            use_enhanced = params.get('enable_forced_alignment', False)
            
            # 使用增强版语音识别函数，支持强制对齐和标点还原
            self.add_print("使用增强版语音识别（强制对齐+标点还原）...")

            def recognition_progress_callback(progress, message=""):
                if message:
                    self.add_print(f"语音识别: {message}")
                elif isinstance(progress, (int, float)):
                    self.add_print(f"语音识别进度: {int(progress)}%")

            recognized = recognize_speech_enhanced(
                audio_path, params['model'],
                detected_language=src_lang,
                device_choice=params['device'],
                progress_callback=recognition_progress_callback,
                word_timestamps=True,
                cd_params=cd_params,
                enable_alignment=params.get('enable_forced_alignment', False)
            )
            
            self.add_print(f"语音识别完成，语言: {recognized['language']}")
            if use_enhanced:
                seg_count = len(recognized.get('segments', []))
                self.add_print(f"生成 {seg_count} 个对齐段落")
            
            recognized_serializable = convert_numpy(recognized)
            
            # 保存强制对齐后的语音识别结果到输出文件夹
            try:
                base_name = os.path.splitext(os.path.basename(video_file))[0]
                aligned_recognition_path = os.path.join(OUTPUT_DIR, f"{base_name}_aligned_recognition.json")
                import json
                def remove_printits_recursive(obj):
                    if isinstance(obj, dict):
                        obj.pop('original_printits', None)
                        obj.pop('_token_count', None)
                        for v in obj.values():
                            remove_printits_recursive(v)
                    elif isinstance(obj, list):
                        for item in obj:
                            remove_printits_recursive(item)
                    return obj
                recognized_for_save = remove_printits_recursive(recognized_serializable)
                with open(aligned_recognition_path, 'w', encoding='utf-8') as f:
                    json.dump(recognized_for_save, f, ensure_ascii=False, indent=2)
                self.add_print(f"强制对齐后的语音识别结果已保存: {aligned_recognition_path}")
            except Exception as e:
                self.add_print(f"保存强制对齐后的语音识别结果失败: {str(e)}")
            
            clear_model_cache()
            self._cleanup(params['device'])

            if self._check_cancelled():
                self.add_print("处理已被用户取消")
                return False, "处理已取消", None, self.prints

            # 3. 翻译
            translated = recognized_serializable
            actual_src_lang = src_lang
            if actual_src_lang != tgt_lang:
                print("[阶段] 3. 翻译")
                self.add_print("3. 翻译...")

                def translation_progress_callback(progress):
                    if isinstance(progress, (int, float)):
                        self.add_print(f"翻译进度: {int(progress)}%")

                try:
                    translated = translate_text(
                        recognized_serializable, translator,
                        target_language=tgt_lang,
                        trans_params=trans_params,
                        server_params=server_params,
                        progress_callback=translation_progress_callback
                    )
                    self.add_print("翻译完成")
                except Exception as e:
                    if "out of memory" in str(e).lower():
                        self.add_print("[内存] 不足，减小上下文窗口重试")
                        retry_params = TransParams.from_dict(params)
                        retry_server_params = ServerParams.from_dict(params)
                        retry_server_params.ctx_size = retry_server_params.ctx_size // 2
                        translated = translate_text(
                            recognized_serializable, translator,
                            target_language=tgt_lang,
                            trans_params=retry_params,
                            server_params=retry_server_params,
                            progress_callback=translation_progress_callback
                        )
                    else:
                        raise
            else:
                self.add_print(f"[跳过翻译] 源语言与目标语言相同 ({actual_src_lang})，将直接使用原文字幕")
            
            self._cleanup(params['device'])

            if self._check_cancelled():
                self.add_print("处理已被用户取消")
                return False, "处理已取消", None, self.prints
            
            # 4. 生成字幕
            print("[阶段] 4. 生成字幕")
            self.add_print("4. 生成字幕...")
            
            # 生成原文版本字幕
            original_subtitle_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(os.path.basename(video_file))[0]}_original_subtitles.srt")
            actual_original_path = generate_subtitle(
                recognized_serializable,
                original_subtitle_path
            )
            self.add_print(f"原文字幕生成完成: {actual_original_path}")

            # 生成译文版本字幕
            translated_subtitle_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(os.path.basename(video_file))[0]}_translated_subtitles.srt")
            actual_translated_path = generate_translated_subtitle(
                translated,
                translated_subtitle_path
            )
            self.add_print(f"译文字幕生成完成: {actual_translated_path}")

            # 生成双语版本字幕
            bilingual_subtitle_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(os.path.basename(video_file))[0]}_bilingual_subtitles.srt")
            actual_bilingual_path = generate_bilingual_subtitle(
                translated,
                bilingual_subtitle_path
            )
            self.add_print(f"双语字幕生成完成: {actual_bilingual_path}")
            
            self.add_print(f"字幕生成完成，总耗时: {time.time() - start:.2f}s")
            
            all_outputs = [original_subtitle_path, translated_subtitle_path, bilingual_subtitle_path]
            all_outputs = [f for f in all_outputs if os.path.exists(f)]

            if all_outputs:
                return True, "处理成功！", all_outputs, self.prints
            missing = [f for f in [original_subtitle_path, translated_subtitle_path, bilingual_subtitle_path] if not os.path.exists(f)]
            return False, f"字幕文件未生成，缺失: {missing}", None, self.prints
            
        except Exception as e:
            import traceback
            print(f"[错误] 处理失败: {e}\n{traceback.format_exc()}")
            self.add_print(f"处理失败: {e}")
            return False, f"处理错误: {e}", None, self.prints
        finally:
            for f in temp_files:
                try:
                    if os.path.exists(f):
                        os.remove(f)
                except Exception:
                    print(f"[清理] 临时文件删除失败: {f}")
                    pass
            self._cleanup(params.get('device', 'cpu'))
    
    def process_queue(self):
        print(f"\n{'='*80}\n[队列] 开始处理，共 {len(self.video_queue)} 个文件")
        
        if not self.video_queue:
            yield [], "", "", 0, ""
            return
        
        if self.processing:
            yield [[i['filename'], i['status']] for i in self.video_queue], "", "", 0, ""
            return
        
        self.processing = True
        self._cancel_flag = False
        
        try:
            for i, item in enumerate(self.video_queue):
                with self._lock:
                    self.video_queue[i]['status'] = '处理中'
                print(f"[队列] 处理第 {i+1}/{len(self.video_queue)} 个: {item['filename']}")
                yield [[j['filename'], j['status']] for j in self.video_queue], "", "", 0, ""
                
                self.prints = []
                success, msg, output, prints = self.process_video(item['file_path'], item['params'])
                print(f"[队列] 结果: {msg}")

                if self._cancel_flag:
                    self._cancel_flag = False
                    break

                with self._lock:
                    self.video_queue[i]['status'] = '已完成' if success else '失败'
                yield [[j['filename'], j['status']] for j in self.video_queue], "", "\n".join(prints), 100, ""
                
                gc.collect()
                time.sleep(0.5)
            
            print(f"[队列] 处理完成，共 {len(self.video_queue)} 个文件")
            yield [[i['filename'], i['status']] for i in self.video_queue], "", "\n".join(self.prints), 100, ""
            
        except Exception as e:
            print(f"[错误] 队列处理失败: {e}")
            yield [], "", "", 0, ""
        
        finally:
            self.processing = False
            self.prints = []
            clear_model_cache()
            clear_translator_cache()
