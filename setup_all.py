import os
import sys
import subprocess

print("=" * 80)
print("Complete Setup - Subtitle Translation Tool")
print("=" * 80)
print()

# Step 1: Setup FFmpeg
print("Step 1: Setting up FFmpeg...")
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

# Step 2: Setup aria2
print("Step 2: Setting up aria2...")
print("-" * 80)

try:
    sys.path.insert(0, os.getcwd())
    from download_all_models import check_aria2_installed, find_aria2_path
    
    aria2_ok = check_aria2_installed()
    aria2_path = find_aria2_path()
    
    if aria2_ok and aria2_path:
        print(f"[OK] aria2 found at: {aria2_path}")
        
        # Test aria2
        try:
            if aria2_path == "aria2c":
                result = subprocess.run(["aria2c", "--version"], capture_output=True, text=True)
            else:
                result = subprocess.run([aria2_path, "--version"], capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"[OK] aria2 version: {result.stdout.splitlines()[0]}")
        except Exception as e:
            print(f"[WARN] {e}")
    else:
        print("[WARN] aria2 not available")
except Exception as e:
    print(f"[ERROR] Failed to setup aria2: {e}")

print()

# Step 3: Verify models
print("Step 3: Checking models...")
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

# Step 4: Summary
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
