# -*- coding: utf-8 -*-
import os


def find_aria2c():
    """Search for aria2c.exe in the project directory"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    possible_paths = [
        os.path.join(os.getcwd(), "aria2c.exe"),
        os.path.join(os.getcwd(), "aria2-1.37.0-win-64bit-build1", "aria2c.exe"),
        os.path.join(os.getcwd(), "aria2", "aria2c.exe"),
        os.path.join(os.getcwd(), "aria2", "aria2-1.37.0-win-64bit-build1", "aria2c.exe"),
        os.path.join(project_root, "aria2c.exe"),
        os.path.join(project_root, "aria2-1.37.0-win-64bit-build1", "aria2c.exe"),
        os.path.join(project_root, "aria2", "aria2c.exe"),
        os.path.join(project_root, "aria2", "aria2-1.37.0-win-64bit-build1", "aria2c.exe"),
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path

    for root, dirs, files in os.walk(os.getcwd()):
        if "aria2c.exe" in files:
            return os.path.join(root, "aria2c.exe")

    for root, dirs, files in os.walk(project_root):
        if "aria2c.exe" in files:
            return os.path.join(root, "aria2c.exe")

    return None


def download_with_progress(url, save_path, max_retries=3):
    """Download file with progress bar, resume support and retries

    Args:
        url: Download URL
        save_path: Local save path
        max_retries: Maximum retry attempts

    Returns:
        bool: Whether download succeeded
    """
    import requests
    from tqdm import tqdm

    for retry in range(max_retries):
        try:
            resume_header = {}
            file_size = 0
            if os.path.exists(save_path):
                file_size = os.path.getsize(save_path)
                if file_size > 0:
                    resume_header['Range'] = f'bytes={file_size}-'
                    print(f"继续下载，已下载 {file_size / (1024*1024):.2f} MB")

            response = requests.get(url, stream=True, headers=resume_header, timeout=60)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            if 'Content-Range' in response.headers:
                content_range = response.headers['Content-Range']
                total_size = int(content_range.split('/')[-1])

            block_size = 1024 * 1024

            mode = 'ab' if file_size > 0 else 'wb'
            with open(save_path, mode) as f, tqdm(
                desc=os.path.basename(save_path),
                total=total_size,
                initial=file_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for data in response.iter_content(block_size):
                    size = f.write(data)
                    bar.update(size)

            return True

        except Exception as e:
            print(f"下载出错 (尝试 {retry+1}/{max_retries}): {str(e)}")
            if retry < max_retries - 1:
                import time
                time.sleep(120)
            else:
                return False

    return False


def verify_zip(zip_path):
    """Verify a ZIP file is valid

    Args:
        zip_path: Path to ZIP file

    Returns:
        bool: Whether the ZIP file is valid
    """
    import zipfile
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.testzip()
        return True
    except zipfile.BadZipFile:
        return False
    except Exception:
        return False


def download_and_extract_zip(url, extract_dir, zip_filename=None, max_retries=3):
    """Download a ZIP file and extract it

    Args:
        url: Download URL
        extract_dir: Directory to extract to
        zip_filename: Name for the temporary ZIP file (default: based on URL)
        max_retries: Maximum retry attempts

    Returns:
        bool: Whether the operation succeeded
    """
    import zipfile

    os.makedirs(extract_dir, exist_ok=True)

    if zip_filename is None:
        zip_filename = url.split('/')[-1]
    zip_path = os.path.join(extract_dir, zip_filename)

    if not download_with_progress(url, zip_path, max_retries):
        return False

    if not verify_zip(zip_path):
        print("ZIP file is corrupted")
        if os.path.exists(zip_path):
            os.remove(zip_path)
        return False

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        os.remove(zip_path)
        return True
    except Exception as e:
        print(f"解压失败: {str(e)}")
        if os.path.exists(zip_path):
            os.remove(zip_path)
        return False


def check_aria2_available(aria2c_path):
    """Check if aria2 is available

    Args:
        aria2c_path: Path to aria2c executable

    Returns:
        tuple: (is_available, message)
    """
    import subprocess

    if not aria2c_path or not os.path.exists(aria2c_path):
        return False, "未找到aria2c.exe"

    try:
        result = subprocess.run([aria2c_path, "--version"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            version = result.stdout.split('\n')[0]
            return True, version
        else:
            return False, f"aria2c执行失败: {result.stderr}"
    except Exception as e:
        return False, f"检查aria2失败: {str(e)}"
