import os
import sys
import urllib.request
import zipfile
import shutil

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
    print(f"Trying source: {source_name}")
    print(f"Download URL: {url}")
    print()
    
    zip_path = os.path.join(ffmpeg_dir, "ffmpeg.zip")
    
    try:
        def report_hook(count, block_size, total_size):
            if total_size > 0:
                percent = int(count * block_size * 100 / total_size)
                sys.stdout.write(f"\rProgress: {percent}%")
                sys.stdout.flush()
        
        urllib.request.urlretrieve(url, zip_path, report_hook)
        print("\n[OK] Download complete!")
        
        if os.path.exists(zip_path):
            print("Extracting...")
            
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
                print(f"[OK] Renamed to: {dest_dir}")
            
            success = True
            break
            
    except Exception as e:
        print(f"\n[ERROR] {e}")
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
