# -*- coding: utf-8 -*-
"""
语音识别模块
包含基础语音识别和强制对齐功能

"""

import os
import gc
try:
    import torch
except ImportError:
    torch = None
import warnings

from config import MODEL_CACHE_DIR, CdParams
from utils.forced_aligner import ForcedAligner



# 公共工具函数


def check_local_model(model_name):
    """检查本地模型文件是否存在"""
    model_paths = [
        os.path.join(MODEL_CACHE_DIR, model_name),
        os.path.join(MODEL_CACHE_DIR, f"openai--whisper-{model_name}"),
        os.path.join(MODEL_CACHE_DIR, f"models--openai--whisper-{model_name}", "snapshots"),
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            if "snapshots" in path and os.path.isdir(path):
                try:
                    snapshot_dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
                    if snapshot_dirs:
                        return os.path.join(path, snapshot_dirs[0])
                except Exception:
                    print(f"[模型检查] 列出快照目录失败: {path}")
                    pass
            elif os.path.isdir(path):
                return path
    
    return None




def _normalize_segment(segment):
    """规范化单个segment，兼容dict和object两种模式"""
    if isinstance(segment, dict):
        start = segment.get('start', 0)
        end = segment.get('end', 0)
        text = segment.get('text', '')
        words = segment.get('words', [])
        chars = segment.get('chars', [])
    else:
        start = getattr(segment, 'start', 0)
        end = getattr(segment, 'end', 0)
        text = getattr(segment, 'text', '')
        words = getattr(segment, 'words', [])
        chars = getattr(segment, 'chars', [])

    processed_words = []
    for word in words:
        if isinstance(word, dict):
            processed_words.append({
                'word': word.get('word', ''),
                'start': word.get('start', 0),
                'end': word.get('end', 0)
            })
        else:
            processed_words.append({
                'word': getattr(word, 'word', ''),
                'start': getattr(word, 'start', 0),
                'end': getattr(word, 'end', 0)
            })

    processed_chars = []
    for char in chars:
        if isinstance(char, dict):
            char_data = {
                'char': char.get('char', ''),
                'start': char.get('start', 0),
                'end': char.get('end', 0)
            }
            if 'original_logits' in char:
                logits_val = char.get('original_logits', 0)
                if hasattr(logits_val, 'flatten'):
                    logits_val = float(logits_val.flatten()[0]) if logits_val.size > 0 else 0.0
                elif hasattr(logits_val, 'item'):
                    logits_val = logits_val.item()
                elif hasattr(logits_val, '__float__'):
                    logits_val = float(logits_val)
                char_data['original_logits'] = logits_val
            processed_chars.append(char_data)
        else:
            char_data = {
                'char': getattr(char, 'char', ''),
                'start': getattr(char, 'start', 0),
                'end': getattr(char, 'end', 0)
            }
            if hasattr(char, 'original_logits'):
                logits_val = getattr(char, 'original_logits', 0)
                if hasattr(logits_val, 'flatten'):
                    logits_val = float(logits_val.flatten()[0]) if logits_val.size > 0 else 0.0
                elif hasattr(logits_val, 'item'):
                    logits_val = logits_val.item()
                elif hasattr(logits_val, '__float__'):
                    logits_val = float(logits_val)
                char_data['original_logits'] = logits_val
            processed_chars.append(char_data)

    return start, end, text, processed_words, processed_chars


def _extract_segment_texts(cd_result):
    """从CD结果中提取片段文本"""
    segments = []

    for segment in cd_result.get('segments', []):
        text = segment.get('text', '') if isinstance(segment, dict) else getattr(segment, 'text', '')
        if not text or not text.strip():
            continue

        start, end, text, processed_words, processed_chars = _normalize_segment(segment)

        segments.append({
            'start': start,
            'end': end,
            'text': text,
            'words': processed_words,
            'chars': processed_chars,
            'language': cd_result.get('language', ''),
            'temperature': segment.get('temperature', 0.0),
            'avg_logprob': segment.get('avg_logprob', 0.0),
            'compression_ratio': segment.get('compression_ratio', 1.0),
            'no_speech_prob': segment.get('no_speech_prob', 0.0)
        })

    print(f"[语音识别] CD结果处理完成，提取 {len(segments)} 个段落")
    return segments


def _apply_forced_alignment(segments, audio_path, language, device):
    """应用强制对齐"""
    if not segments:
        return segments

    aligner = None
    try:
        print("[强制对齐] 启用强制对齐...")
        aligner = ForcedAligner(device=device)
        if aligner.load_alignment_model(language):
            segments = aligner.align(segments, audio_path, return_char_alignments=True)
            print("[强制对齐] 强制对齐完成")
        else:
            print("[强制对齐] 强制对齐模型加载失败，跳过对齐")
    except Exception as e:
        print(f"[强制对齐] 强制对齐失败: {str(e)}")
    finally:
        if aligner is not None:
            aligner.cleanup()
            print("[内存管理] 强制对齐模型已卸载，清理显存")
            if torch is not None and torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"[内存管理] 强制对齐卸载后 GPU 状态: 已分配 {allocated:.2f}GB, 已保留 {reserved:.2f}GB")

    return segments


def _build_final_segments(segments, audio_path, language=''):
    """构建最终输出片段"""
    result = {
        'segments': segments,
        'text': '',
        'language': language
    }

    for segment in segments:
        text = segment.get('text', '')
        if not result['text']:
            result['text'] = text
        else:
            if language.startswith(('ja', 'zh', 'ko')):
                result['text'] += text
            else:
                result['text'] += ' ' + text

    return result


def _process_cd_segments(cd_result, audio_path, language=None, device="auto", enable_alignment=True):
    """处理 Whisper-CD 结果的共享函数"""
    detected_language = language or cd_result.get('language', '')
    segments = _extract_segment_texts(cd_result)

    if enable_alignment:
        segments = _apply_forced_alignment(segments, audio_path, detected_language or 'ja', device)

    result = _build_final_segments(segments, audio_path, detected_language)
    return result


def _cleanup_memory():
    """清理内存和显存"""
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def recognize_speech_enhanced(audio_path, model_path, detected_language=None, device_choice="auto",
                    progress_callback=None, word_timestamps=True,
                    cd_params: CdParams = None,
                    enable_alignment=True):
    """增强版语音识别

    Args:
        audio_path: 音频文件路径
        model_path: Whisper模型名称或路径
        detected_language: 检测到的语言
        device_choice: 设备选择
        progress_callback: 进度回调
        word_timestamps: 是否启用单词时间戳
        cd_params: CdParams对象，包含所有对比解码参数
        enable_alignment: 是否启用强制对齐

    Returns:
        识别结果字典
    """
    print("[语音识别] 使用增强版语音识别器进行语音识别")

    if cd_params is None:
        cd_params = CdParams()

    device = "cuda" if device_choice != "cpu" else "cpu"

    from config import config
    from utils.whisper_cd_original import WhisperCDOriginal

    print("[Whisper-CD] 启用 Whisper-CD 处理器...")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        whispercd_processor = WhisperCDOriginal(
            model_path=model_path,
            device=device,
            cd_params=cd_params,
            enable_alignment=enable_alignment,
        )

        print("[Whisper-CD] 应用对比解码...")
        cd_result = whispercd_processor.transcribe(
            audio_path,
            detected_language,
            progress_callback=progress_callback
        )

    whispercd_processor.cleanup()
    del whispercd_processor
    _cleanup_memory()

    if torch is not None and torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[内存管理] Whisper-CD 卸载后 GPU 状态: 已分配 {allocated:.2f}GB, 已保留 {reserved:.2f}GB")

    if progress_callback:
        progress_callback(80)

    result = _process_cd_segments(cd_result, audio_path, detected_language, device, enable_alignment)

    print("[内存管理] 转录完成，执行最终内存清理...")
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[内存管理] 最终清理后 GPU 状态: 已分配 {allocated:.2f}GB, 已保留 {reserved:.2f}GB")

    return result


def clear_model_cache():
    """清理模型缓存"""
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[缓存] 模型缓存已清理，GPU: 已分配 {allocated:.2f}GB, 已保留 {reserved:.2f}GB")
