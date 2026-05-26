# -*- coding: utf-8 -*-
import os
import sys

# 强制使用 UTF-8 编码
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['LC_ALL'] = 'en_US.UTF-8'
os.environ['LANG'] = 'en_US.UTF-8'

import subprocess

print("=" * 80)
print("Complete Setup - Subtitle Translation Tool")
print("=" * 80)
print()

# Step 1: Setup aria2
print("Step 1: Setting up aria2...")
print("-" * 80)

try:
    sys.path.insert(0, os.getcwd())
    from aria2_downloader import Aria2Downloader
    
    # 创建下载器实例
    downloader = Aria2Downloader()
    
    # 检查aria2是否可用
    is_available, msg = downloader.check_aria2()
    
    if is_available:
        print(f"[OK] aria2 is available: {msg}")
    else:
        print(f"[INFO] aria2 not found: {msg}")
        print("[INFO] Downloading aria2...")
        
        # 下载aria2
        success, download_msg = downloader.download_aria2()
        if success:
            print(f"[OK] aria2 downloaded successfully: {download_msg}")
        else:
            print(f"[WARN] aria2 download failed: {download_msg}")
            print("[INFO] aria2 is required for faster downloads")
except Exception as e:
    print(f"[ERROR] Failed to setup aria2: {e}")

print()

# Step 2: Setup FFmpeg
print("Step 2: Setting up FFmpeg...")
print("-" * 80)

ffmpeg_dir = os.path.join(os.getcwd(), "ffmpeg")
ffmpeg_exe = None

if os.path.exists(ffmpeg_dir):
    for root, dirs, files in os.walk(ffmpeg_dir):
        for file in files:
            if file == "ffmpeg.exe":
                ffmpeg_exe = os.path.join(root, file)
                break
        if ffmpeg_exe:
            break

if ffmpeg_exe:
    print(f"[OK] FFmpeg found at: {ffmpeg_exe}")
else:
    print("[INFO] FFmpeg not found, downloading...")
    
    # Try to run the Python download script
    download_ffmpeg_script = os.path.join(os.getcwd(), "download_ffmpeg.py")
    if os.path.exists(download_ffmpeg_script):
        print("Running FFmpeg download script...")
        try:
            subprocess.run([sys.executable, download_ffmpeg_script], check=True)
        except Exception as e:
            print(f"[WARN] FFmpeg download failed: {e}")
    else:
        print("[WARN] download_ffmpeg.py not found")

print()

# Step 3: Setup llama.cpp
print("Step 3: Setting up llama.cpp...")
print("-" * 80)

llama_cpp_dir = os.path.join(os.getcwd(), "llama_cpp")
llama_cli_exe = os.path.join(llama_cpp_dir, "llama-cli.exe")

if os.path.exists(llama_cli_exe):
    print(f"[OK] llama.cpp found at: {llama_cli_exe}")
else:
    print("[INFO] llama.cpp not found, downloading...")
    
    # Run the Python download script
    download_llama_cpp_script = os.path.join(os.getcwd(), "download_llama_cpp.py")
    if os.path.exists(download_llama_cpp_script):
        print("Running llama.cpp download script...")
        try:
            subprocess.run([sys.executable, download_llama_cpp_script], check=True)
        except Exception as e:
            print(f"[WARN] llama.cpp download failed: {e}")
    else:
        print("[WARN] download_llama_cpp.py not found")

print()

# Step 4: Verify models
print("Step 4: Checking models...")
print("-" * 80)

models_dir = os.path.join(os.getcwd(), "models")
if os.path.exists(models_dir):
    model_count = 0
    print("Available models:")
    for item in os.listdir(models_dir):
        item_path = os.path.join(models_dir, item)
        if os.path.isdir(item_path):
            model_count += 1
            print(f"  [{model_count}] {item}")
    print()
    print(f"[OK] Found {model_count} models")
else:
    print("[INFO] Models directory not found")
    print()
    print("To download models, run:")
    print("  python download_all_models.py")

print()

# Step 5: Summary
print("=" * 80)
print("SETUP COMPLETE")
print("=" * 80)
print()
print("Next steps:")
print()
print("1. Download models (if needed):")
print("   python download_all_models.py")
print()
print("2. Run the UI:")
print("   python ui.py")
print()
print("=" * 80)
