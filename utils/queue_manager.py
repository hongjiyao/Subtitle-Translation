#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""队列管理器模块"""
import os
import time
import gc
import threading
try:
    import torch
except ImportError:
    torch = None



from config import MODEL_OPTIONS, TEMP_DIR, OUTPUT_DIR, config, CdParams, TransParams, ServerParams, PARAM_DEFINITIONS
from utils.video_processor import extract_audio
from utils.speech_recognizer import recognize_speech_enhanced, clear_model_cache
from utils.translator import translate_text, clear_translator_cache
from utils.subtitle_generator import generate_subtitle, generate_translated_subtitle, generate_bilingual_subtitle

VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.mpg', '.mpeg', '.ts']
MAX_FILE_SIZE = 10 * 1024 * 1024 * 1024

def _cleanup_gpu_memory():
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()

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


class VideoProcessorPipeline:
    """视频处理管线：负责单个视频的完整处理流程"""

    def __init__(self, add_print_callback=None, check_cancelled_fn=None, cleanup_fn=None):
        self._add_print = add_print_callback or (lambda msg: None)
        self._check_cancelled = check_cancelled_fn or (lambda: False)
        self._cleanup = cleanup_fn or (lambda device='cpu': None)

    def _step_extract_audio(self, video_path, output_dir, progress_cb):
        print("[阶段] 1. 提取音频")
        progress_cb("1. 提取音频...")
        start = time.time()
        audio_path = extract_audio(video_path)
        progress_cb(f"音频提取完成，耗时: {time.time() - start:.2f}s")
        return audio_path, start

    def _step_recognize(self, audio_path, config, progress_cb):
        print("[阶段] 2. 语音识别")
        progress_cb("2. 语音识别...")

        use_enhanced = config.get('enable_forced_alignment', False)

        progress_cb("使用增强版语音识别（强制对齐+标点还原）...")

        def recognition_progress_callback(progress, message=""):
            if message:
                progress_cb(f"语音识别: {message}")
            elif isinstance(progress, (int, float)):
                progress_cb(f"语音识别进度: {int(progress)}%")

        recognized = recognize_speech_enhanced(
            audio_path, config.get('model', 'large-v3'),
            detected_language=config.get('src_lang', 'en'),
            device_choice=config.get('device', 'auto'),
            progress_callback=recognition_progress_callback,
            word_timestamps=True,
            cd_params=config.get('cd_params', {}),
            enable_alignment=config.get('enable_forced_alignment', False)
        )

        progress_cb(f"语音识别完成，语言: {recognized.get('language', 'en')}")
        if use_enhanced:
            seg_count = len(recognized.get('segments', []))
            progress_cb(f"生成 {seg_count} 个对齐段落")

        recognized_serializable = convert_numpy(recognized)

        try:
            base_name = os.path.splitext(os.path.basename(config.get('video_file', '')))[0]
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
            progress_cb(f"强制对齐后的语音识别结果已保存: {aligned_recognition_path}")
        except Exception as e:
            progress_cb(f"保存强制对齐后的语音识别结果失败: {str(e)}")

        clear_model_cache()
        self._cleanup(config.get('device', 'auto'))

        return recognized_serializable

    def _step_translate(self, segments, config, progress_cb):
        translated = segments
        actual_src_lang = config.get('src_lang', 'en')
        tgt_lang = config.get('tgt_lang', 'zh')
        if actual_src_lang != tgt_lang:
            print("[阶段] 3. 翻译")
            progress_cb("3. 翻译...")

            def translation_progress_callback(progress):
                if isinstance(progress, (int, float)):
                    progress_cb(f"翻译进度: {int(progress)}%")

            try:
                translated = translate_text(
                    segments, config.get('translator', 'llama-server'),
                    target_language=tgt_lang,
                    trans_params=config.get('trans_params', TransParams()),
                    server_params=config.get('server_params', ServerParams()),
                    progress_callback=translation_progress_callback
                )
                progress_cb("翻译完成")
            except Exception as e:
                if "out of memory" in str(e).lower():
                    progress_cb("[内存] 不足，减小参数重试")
                    retry_params = TransParams.from_dict(config.get('params', {}))
                    retry_server_params = ServerParams.from_dict(config.get('params', {}))
                    retry_server_params.ctx_size = max(1024, retry_server_params.ctx_size // 2)
                    retry_server_params.batch_size = max(256, retry_server_params.batch_size // 2)
                    retry_params.max_output_tokens = max(64, retry_params.max_output_tokens // 2)
                    translated = translate_text(
                        segments, config.get('translator', 'llama-server'),
                        target_language=tgt_lang,
                        trans_params=retry_params,
                        server_params=retry_server_params,
                        progress_callback=translation_progress_callback
                    )
                else:
                    raise
        else:
            progress_cb(f"[跳过翻译] 源语言与目标语言相同 ({actual_src_lang})，将直接使用原文字幕")

        self._cleanup(config.get('device', 'auto'))

        return translated

    def _step_generate_subtitles(self, segments, translated_segments, output_dir, base_name, config):
        print("[阶段] 4. 生成字幕")
        self._add_print("4. 生成字幕...")

        original_subtitle_path = os.path.join(output_dir, f"{base_name}_original_subtitles.srt")
        actual_original_path = generate_subtitle(
            segments,
            original_subtitle_path
        )
        self._add_print(f"原文字幕生成完成: {actual_original_path}")

        translated_subtitle_path = os.path.join(output_dir, f"{base_name}_translated_subtitles.srt")
        actual_translated_path = generate_translated_subtitle(
            translated_segments,
            translated_subtitle_path
        )
        self._add_print(f"译文字幕生成完成: {actual_translated_path}")

        bilingual_subtitle_path = os.path.join(output_dir, f"{base_name}_bilingual_subtitles.srt")
        actual_bilingual_path = generate_bilingual_subtitle(
            translated_segments,
            bilingual_subtitle_path
        )
        self._add_print(f"双语字幕生成完成: {actual_bilingual_path}")

        self._add_print(f"字幕生成完成，总耗时: {time.time() - config.get('start', time.time()):.2f}s")

        all_outputs = [original_subtitle_path, translated_subtitle_path, bilingual_subtitle_path]
        all_outputs = [f for f in all_outputs if os.path.exists(f)]

        if all_outputs:
            return True, "处理成功！", all_outputs
        missing = [f for f in [original_subtitle_path, translated_subtitle_path, bilingual_subtitle_path] if not os.path.exists(f)]
        return False, f"字幕文件未生成，缺失: {missing}", None

    def process_video(self, video_file, params):
        temp_files = []
        try:
            if not video_file or not os.path.exists(video_file):
                return False, "文件不存在", None, []

            if os.path.splitext(video_file)[1].lower() not in VIDEO_EXTENSIONS:
                return False, "不支持的格式", None, []

            if params.get('model', 'large-v3') not in MODEL_OPTIONS:
                return False, "模型选择错误", None, []

            translator = params.get('translator', 'llama-server')
            if translator not in PARAM_DEFINITIONS['translator']['options']:
                return False, f"翻译模型错误: {translator} 不在支持的模型列表中", None, []

            output_path = params.get('output_path') or os.path.join(OUTPUT_DIR, f"{os.path.splitext(os.path.basename(video_file))[0]}_subtitles.srt")
            src_lang = params.get('source_language', 'ja')
            tgt_lang = params.get('target_language', 'zh')
            cd_params = CdParams.from_dict(params)
            trans_params = TransParams.from_dict(params)
            server_params = ServerParams.from_dict(params)

            self._add_print(f"开始处理: {video_file}")
            os.makedirs(TEMP_DIR, exist_ok=True)

            audio_path, start = self._step_extract_audio(video_file, TEMP_DIR, self._add_print)
            temp_files.append(audio_path)

            if self._check_cancelled():
                self._add_print("处理已被用户取消")
                return False, "处理已取消", None, None

            recognize_config = {
                'model': params.get('model', 'large-v3'),
                'src_lang': src_lang,
                'device': params.get('device', 'auto'),
                'cd_params': cd_params,
                'enable_forced_alignment': params.get('enable_forced_alignment', False),
                'video_file': video_file,
            }
            recognized_serializable = self._step_recognize(audio_path, recognize_config, self._add_print)

            if self._check_cancelled():
                self._add_print("处理已被用户取消")
                return False, "处理已取消", None, None

            translate_config = {
                'translator': translator,
                'tgt_lang': tgt_lang,
                'src_lang': src_lang,
                'trans_params': trans_params,
                'server_params': server_params,
                'device': params.get('device', 'auto'),
                'params': params,
            }
            translated = self._step_translate(recognized_serializable, translate_config, self._add_print)

            if self._check_cancelled():
                self._add_print("处理已被用户取消")
                return False, "处理已取消", None, None

            base_name = os.path.splitext(os.path.basename(video_file))[0]
            subtitle_config = {
                'start': start,
            }
            success, msg, outputs = self._step_generate_subtitles(
                recognized_serializable, translated, OUTPUT_DIR, base_name, subtitle_config
            )

            if success:
                return True, msg, outputs, None
            return False, msg, outputs, None

        except Exception as e:
            import traceback
            print(f"[错误] 处理失败: {e}\n{traceback.format_exc()}")
            self._add_print(f"处理失败: {e}")
            return False, f"处理错误: {e}", None, None
        finally:
            for f in temp_files:
                try:
                    if os.path.exists(f):
                        os.remove(f)
                except Exception:
                    print(f"[清理] 临时文件删除失败: {f}")
                    pass
            self._cleanup(params.get('device', 'cpu'))


class QueueManager:
    def __init__(self):
        self.video_queue = []
        self.processing = False
        self.prints = []
        self._lock = threading.Lock()
        self._cancel_event = threading.Event()
        self._pipeline = VideoProcessorPipeline(
            add_print_callback=self.add_print,
            check_cancelled_fn=self._check_cancelled,
            cleanup_fn=self._cleanup,
        )
    
    def cancel_processing(self):
        self._cancel_event.set()
        print("[队列] 收到取消请求")

    def _check_cancelled(self):
        if self._cancel_event.is_set():
            self._cancel_event.clear()
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
        _cleanup_gpu_memory()

    def process_video(self, video_file, params):
        success, msg, outputs, _ = self._pipeline.process_video(video_file, params)
        return success, msg, outputs, self.prints
    
    def process_queue(self):
        print(f"\n{'='*80}\n[队列] 开始处理，共 {len(self.video_queue)} 个文件")
        
        if not self.video_queue:
            yield [], "", "", 0, ""
            return
        
        if self.processing:
            yield [[i['filename'], i['status']] for i in self.video_queue], "", "", 0, ""
            return
        
        self.processing = True
        self._cancel_event.clear()
        
        try:
            for i, item in enumerate(self.video_queue):
                with self._lock:
                    self.video_queue[i]['status'] = '处理中'
                print(f"[队列] 处理第 {i+1}/{len(self.video_queue)} 个: {item['filename']}")
                yield [[j['filename'], j['status']] for j in self.video_queue], "", "", 0, ""
                
                self.prints = []
                success, msg, output, prints = self.process_video(item['file_path'], item['params'])
                print(f"[队列] 结果: {msg}")

                if self._cancel_event.is_set():
                    self._cancel_event.clear()
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
