import subprocess
import os
import glob
from config import TEMP_DIR

def find_ffmpeg():
    """查找FFmpeg可执行文件"""
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
    audio_path = os.path.join(TEMP_DIR, "audio.wav")
    
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
