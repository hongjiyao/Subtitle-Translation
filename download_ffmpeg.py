# -*- coding: utf-8 -*-
import os
import sys
import subprocess
import zipfile
import shutil
import time

from utils.logger import log_message
from utils.download_utils import download_with_progress, verify_zip



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
        if download_with_progress(url, zip_path):
            print("\n[OK] Download complete!")
            
            if not verify_zip(zip_path):
                log_message("ZIP file is corrupted", "ERROR")
                if os.path.exists(zip_path):
                    os.remove(zip_path)
                log_message("Trying next source...", "INFO")
                print()
                continue
            log_message("ZIP file verification passed", "INFO")
        else:
            log_message("Download failed", "ERROR")
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
