# -*- coding: utf-8 -*-
import os
import sys

# 强制使用 UTF-8 编码
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['LC_ALL'] = 'en_US.UTF-8'
os.environ['LANG'] = 'en_US.UTF-8'

import subprocess
import zipfile
import shutil
import time

def log_message(message, level="INFO"):
    """记录日志信息"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    
    # 根据级别设置前缀
    if level == "INFO":
        prefix = "[INFO]"
    elif level == "ERROR":
        prefix = "[ERROR]"
    elif level == "SUCCESS":
        prefix = "[SUCCESS]"
    elif level == "WARN":
        prefix = "[WARN]"
    else:
        prefix = "[INFO]"
    
    print(f"{timestamp} {prefix} {message}")



print("=" * 70)
print("Downloading FFmpeg")
print("=" * 70)
print()

ffmpeg_dir = os.path.join(os.getcwd(), "ffmpeg")
os.makedirs(ffmpeg_dir, exist_ok=True)

# Download sources
sources = [
    ("Gyan.dev", "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"),
    ("GitHub", "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"),
    
]

success = False

for source_name, url in sources:
    log_message(f"Trying source: {source_name}")
    log_message(f"Download URL: {url}")
    print()
    
    zip_path = os.path.join(ffmpeg_dir, "ffmpeg.zip")
    
    try:
        import requests
        from tqdm import tqdm
        
        # 检查文件是否已存在且有部分内容
        resume_header = {}
        file_size = 0
        if os.path.exists(zip_path):
            file_size = os.path.getsize(zip_path)
            if file_size > 0:
                resume_header['Range'] = f'bytes={file_size}-'
                log_message(f"继续下载，已下载 {file_size / (1024*1024):.2f} MB", "INFO")
        
        response = requests.get(url, stream=True, headers=resume_header, timeout=60)
        response.raise_for_status()
        
        # 获取总文件大小
        total_size = int(response.headers.get('content-length', 0))
        if 'Content-Range' in response.headers:
            # 处理续传情况
            content_range = response.headers['Content-Range']
            total_size = int(content_range.split('/')[-1])
        
        block_size = 1024 * 1024  # 1MB
        
        # 以追加模式打开文件
        mode = 'ab' if file_size > 0 else 'wb'
        with open(zip_path, mode) as f, tqdm(
            desc=os.path.basename(zip_path),
            total=total_size,
            initial=file_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(block_size):
                size = f.write(data)
                bar.update(size)
        
        print("\n[OK] Download complete!")
        
        # 验证下载的文件是否为有效的ZIP文件
        log_message("Verifying ZIP file...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # 检查ZIP文件是否有效
                zip_ref.testzip()
            log_message("ZIP file verification passed", "INFO")
        except zipfile.BadZipFile as e:
            log_message(f"ZIP file is corrupted: {str(e)}", "ERROR")
            if os.path.exists(zip_path):
                os.remove(zip_path)
            log_message("Trying next source...", "INFO")
            print()
            continue
        except Exception as e:
            log_message(f"Error verifying ZIP file: {str(e)}", "ERROR")
            if os.path.exists(zip_path):
                os.remove(zip_path)
            log_message("Trying next source...", "INFO")
            print()
            continue
    except Exception as e:
        log_message(f"下载失败: {str(e)}", "ERROR")
        log_message("正在重试...", "INFO")
        import time
        time.sleep(120)  # 等待120秒后重试
        # 不要删除部分下载的文件，以便断点续传
        continue
    
    try:
        if os.path.exists(zip_path):
            log_message("Extracting...")
            
            if zip_path.endswith('.zip'):
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(ffmpeg_dir)
            
            os.remove(zip_path)
            
            # Find and rename
            extracted_dirs = [d for d in os.listdir(ffmpeg_dir) if os.path.isdir(os.path.join(ffmpeg_dir, d))]
            if extracted_dirs:
                src_dir = os.path.join(ffmpeg_dir, extracted_dirs[0])
                dest_dir = os.path.join(ffmpeg_dir, "ffmpeg-master-latest-win64-gpl")
                
                if os.path.exists(dest_dir):
                    shutil.rmtree(dest_dir)
                
                shutil.move(src_dir, dest_dir)
                log_message(f"Renamed to: {dest_dir}")
            
            success = True
            break
    except Exception as e:
        log_message(f"Error: {str(e)}", "ERROR")
        if os.path.exists(zip_path):
            os.remove(zip_path)
        log_message("Trying next source...", "INFO")
        print()

print()
print("=" * 70)
if success:
    print("[SUCCESS] FFmpeg installed successfully!")
    sys.exit(0)
else:
    print("[FAIL] FFmpeg installation failed")
    sys.exit(1)
print("=" * 70)
