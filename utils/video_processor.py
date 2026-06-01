# -*- coding: utf-8 -*-
import os
import re
import subprocess
from config import TEMP_DIR

_INVALID_PATH_CHARS_PATTERN = re.compile(r'[\x00-\x1f]')

def _get_allowed_dirs():
    dirs = []
    try:
        from config import PROJECT_ROOT, OUTPUT_DIR
        dirs.append(os.path.realpath(PROJECT_ROOT))
        dirs.append(os.path.realpath(OUTPUT_DIR))
    except ImportError:
        dirs.append(os.path.realpath(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    dirs.append(os.path.realpath(TEMP_DIR))
    dirs.append(os.path.realpath(os.path.expanduser("~")))
    return dirs

_ALLOWED_DIRS = _get_allowed_dirs()

def validate_path(file_path):
    if not file_path or not isinstance(file_path, str):
        raise ValueError("路径不能为空且必须为字符串")
    if _INVALID_PATH_CHARS_PATTERN.search(file_path):
        raise ValueError(f"路径包含非法控制字符: {file_path}")
    resolved = os.path.realpath(file_path)
    if resolved != os.path.normpath(os.path.abspath(file_path)):
        raise ValueError(f"路径解析异常，可能包含路径遍历: {file_path}")
    allowed = any(resolved.startswith(d + os.sep) or resolved == d for d in _ALLOWED_DIRS)
    if not allowed:
        raise ValueError(f"路径不在允许的目录范围内: {file_path}")
    return resolved

def find_ffmpeg():
    try:
        from config import PROJECT_ROOT
        current_dir = PROJECT_ROOT
    except ImportError:
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ffmpeg_dir = os.path.join(current_dir, "ffmpeg")
    
    # 尝试多种可能的FFmpeg目录结构
    possible_paths = [
        os.path.join(ffmpeg_dir, "ffmpeg-master-latest-win64-gpl", "bin", "ffmpeg.exe"),
        os.path.join(ffmpeg_dir, "bin", "ffmpeg.exe"),
    ]
    
    # 查找任意以 ffmpeg 开头的目录
    if os.path.exists(ffmpeg_dir):
        subdirs = [d for d in os.listdir(ffmpeg_dir) if os.path.isdir(os.path.join(ffmpeg_dir, d))]
        for subdir in subdirs:
            possible_paths.append(os.path.join(ffmpeg_dir, subdir, "bin", "ffmpeg.exe"))
    
    # 检查所有可能的路径
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    # 如果都没找到，使用系统FFmpeg
    return "ffmpeg"

def extract_audio(video_path):
    """从视频中提取音频"""
    video_path = validate_path(video_path)
    import uuid
    # 使用唯一标识符生成临时音频文件名，避免文件冲突
    unique_id = str(uuid.uuid4())[:8]
    audio_filename = f"audio_{unique_id}.wav"
    audio_path = os.path.join(TEMP_DIR, audio_filename)
    
    # 查找FFmpeg
    ffmpeg_path = find_ffmpeg()
    
    # 构建FFmpeg命令
    cmd = [
        ffmpeg_path, "-i", video_path,
        "-ac", "1", "-ar", "16000", "-f", "wav",
        "-y", audio_path
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        print(f"音频提取完成: {audio_path}")
        return audio_path
    except subprocess.CalledProcessError as e:
        # 安全处理错误信息
        error_msg = ""
        if e.stderr:
            error_msg = str(e.stderr)
        print(f"音频提取失败: {error_msg}")
        raise
