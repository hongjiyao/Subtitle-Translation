import os
import sys
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

def find_aria2_path():
    """查找aria2可执行文件路径"""
    # 首先检查系统PATH
    try:
        result = subprocess.run(["aria2c", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            return "aria2c"
    except:
        pass
    
    # 检查当前目录下的aria2文件夹
    aria2_dir = os.path.join(os.getcwd(), "aria2")
    if os.path.exists(aria2_dir):
        for root, dirs, files in os.walk(aria2_dir):
            for file in files:
                if file == "aria2c.exe":
                    return os.path.join(root, file)
    
    return None

def get_cpu_core_count():
    """获取CPU核心数"""
    import os
    try:
        return os.cpu_count() or 4  # 默认4核心
    except:
        return 4

def download_with_aria2(url, output_path):
    """使用aria2下载文件（已禁用，直接返回失败）"""
    log_message("aria2 download disabled, falling back to urllib", "INFO")
    return False

print("=" * 70)
print("Downloading FFmpeg")
print("=" * 70)
print()

ffmpeg_dir = os.path.join(os.getcwd(), "ffmpeg")
os.makedirs(ffmpeg_dir, exist_ok=True)

# Download sources
sources = [
    ("GitHub", "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"),
    ("Gyan.dev", "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"),
]

success = False

for source_name, url in sources:
    log_message(f"Trying source: {source_name}")
    log_message(f"Download URL: {url}")
    print()
    
    zip_path = os.path.join(ffmpeg_dir, "ffmpeg.zip")
    
    try:
        # 尝试使用aria2下载
        aria2_success = download_with_aria2(url, zip_path)
        
        # 如果aria2下载失败，回退到urllib
        if not aria2_success:
            log_message("Falling back to urllib download", "WARN")
            import urllib.request
            
            def report_hook(count, block_size, total_size):
                if total_size > 0:
                    percent = int(count * block_size * 100 / total_size)
                    sys.stdout.write(f"\rProgress: {percent}%")
                    sys.stdout.flush()
            
            urllib.request.urlretrieve(url, zip_path, report_hook)
            print("\n[OK] Download complete!")
        
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
        print()

print()
print("=" * 70)
if success:
    print("[SUCCESS] FFmpeg installed successfully!")
else:
    print("[FAIL] FFmpeg installation failed")
print("=" * 70)
