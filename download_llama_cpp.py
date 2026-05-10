#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下载llama.cpp CUDA版本
"""

import os
import sys
import zipfile
import shutil
import argparse

from utils.logger import log_message
from utils.download_utils import download_with_progress, verify_zip

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
LLAMA_CPP_DIR = os.path.join(PROJECT_ROOT, "llama_cpp")

LLAMA_CPP_CUDA_URLS = [
    "https://github.com/ggml-org/llama.cpp/releases/download/b8514/llama-b8514-bin-win-cuda-12.4-x64.zip",
    "https://ghproxy.com/https://github.com/ggml-org/llama.cpp/releases/download/b8514/llama-b8514-bin-win-cuda-12.4-x64.zip"
]


def download_llama_cpp():
    """下载并安装llama.cpp"""
    log_message("开始下载llama.cpp CUDA版本...")
    
    zip_file = os.path.join(PROJECT_ROOT, "llama_cpp_cuda.zip")
    
    downloaded = False
    for i, url in enumerate(LLAMA_CPP_CUDA_URLS, 1):
        log_message(f"尝试下载源 {i}/{len(LLAMA_CPP_CUDA_URLS)}")
        log_message(f"下载链接: {url}")
        log_message(f"正在下载到: {zip_file}")
        
        if download_with_progress(url, zip_file):
            if verify_zip(zip_file):
                log_message("ZIP file verification passed")
                downloaded = True
                break
            else:
                log_message("ZIP file is corrupted")
                if os.path.exists(zip_file):
                    os.remove(zip_file)
                log_message(f"下载源 {i} 失败，尝试下一个源...")
                continue
        else:
            log_message(f"下载源 {i} 失败，尝试下一个源...")
            if os.path.exists(zip_file):
                os.remove(zip_file)
            continue
    
    if not downloaded:
        log_message("所有下载源都失败，退出")
        return False
    
    file_size = os.path.getsize(zip_file)
    log_message(f"下载文件大小: {file_size / (1024 * 1024):.2f} MB")
    
    if file_size < 10 * 1024 * 1024:
        log_message("警告: 下载文件可能不完整")
    
    if not os.path.exists(LLAMA_CPP_DIR):
        os.makedirs(LLAMA_CPP_DIR, exist_ok=True)
        log_message(f"创建目录: {LLAMA_CPP_DIR}")
    else:
        log_message(f"目录已存在: {LLAMA_CPP_DIR}")
        for item in os.listdir(LLAMA_CPP_DIR):
            item_path = os.path.join(LLAMA_CPP_DIR, item)
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
        log_message("清空目录内容")
    
    log_message("正在解压...")
    try:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(LLAMA_CPP_DIR)
        log_message(f"成功解压到: {LLAMA_CPP_DIR}")
    except Exception as e:
        log_message(f"解压失败: {str(e)}")
        return False
    
    try:
        os.remove(zip_file)
        log_message(f"已删除临时下载文件: {zip_file}")
    except Exception as e:
        log_message(f"删除临时文件失败: {str(e)}")
    
    if not os.listdir(LLAMA_CPP_DIR):
        log_message("解压后目录为空")
        return False
    
    llama_cli_path = os.path.join(LLAMA_CPP_DIR, "llama-cli.exe")
    if not os.path.exists(llama_cli_path):
        log_message("警告: 未找到llama-cli.exe")
    else:
        log_message(f"找到llama-cli.exe: {llama_cli_path}")
    
    log_message("llama.cpp CUDA版本下载完成")
    return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="下载llama.cpp CUDA版本")
    parser.parse_args()
    
    log_message("=" * 60)
    log_message("llama.cpp CUDA版本下载工具")
    log_message("=" * 60)
    
    # 检查网络连接
    try:
        import requests
        response = requests.get("https://github.com", timeout=10)
        if response.status_code != 200:
            log_message("警告: 无法访问GitHub")
    except:
        log_message("警告: 网络连接失败")
    
    # 下载llama.cpp
    success = download_llama_cpp()
    
    if success:
        log_message("\n✓ 下载成功")
        log_message(f"llama.cpp目录: {LLAMA_CPP_DIR}")
        return 0
    else:
        log_message("\n✗ 下载失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())
