#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下载llama.cpp CUDA版本
"""

import os
import sys
import requests
import zipfile
import shutil
import tempfile
import argparse
from tqdm import tqdm

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
LLAMA_CPP_DIR = os.path.join(PROJECT_ROOT, "llama_cpp")

# 下载链接列表
LLAMA_CPP_CUDA_URLS = [
    "https://github.com/ggml-org/llama.cpp/releases/download/b8514/llama-b8514-bin-win-cuda-12.4-x64.zip",
    "https://ghproxy.com/https://github.com/ggml-org/llama.cpp/releases/download/b8514/llama-b8514-bin-win-cuda-12.4-x64.zip"
]


def log_message(message):
    """记录日志信息"""
    print(f"[INFO] {message}")


def download_file(url, save_path, max_retries=3):
    """下载文件并显示进度，支持断点续传和重试"""
    for retry in range(max_retries):
        try:
            # 检查文件是否已存在且有部分内容
            resume_header = {}
            file_size = 0
            if os.path.exists(save_path):
                file_size = os.path.getsize(save_path)
                if file_size > 0:
                    resume_header['Range'] = f'bytes={file_size}-'
                    log_message(f"继续下载，已下载 {file_size / (1024*1024):.2f} MB")
            
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
            
            # 验证下载的文件是否为有效的ZIP文件
            try:
                with zipfile.ZipFile(save_path, 'r') as zip_ref:
                    # 检查ZIP文件是否有效
                    zip_ref.testzip()
                log_message("ZIP file verification passed")
                return True
            except zipfile.BadZipFile as e:
                log_message(f"ZIP file is corrupted: {str(e)}")
                if os.path.exists(save_path):
                    os.remove(save_path)
                if retry < max_retries - 1:
                    log_message("正在重试...")
                    import time
                    time.sleep(120)  # 等待120秒后重试
                else:
                    log_message("达到最大重试次数")
                return False
        except requests.exceptions.RequestException as e:
            log_message(f"网络请求失败 (尝试 {retry+1}/{max_retries}): {str(e)}")
            if retry < max_retries - 1:
                log_message("正在重试...")
                import time
                time.sleep(120)  # 等待120秒后重试
            else:
                log_message("达到最大重试次数")
                return False
        except Exception as e:
            log_message(f"下载失败 (尝试 {retry+1}/{max_retries}): {str(e)}")
            if retry < max_retries - 1:
                log_message("正在重试...")
                import time
                time.sleep(120)  # 等待120秒后重试
            else:
                log_message("达到最大重试次数")
                return False


def extract_zip(zip_path, extract_dir):
    """解压zip文件"""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        log_message(f"成功解压到: {extract_dir}")
        return True
    except Exception as e:
        log_message(f"解压失败: {str(e)}")
        return False


def download_llama_cpp():
    """下载并安装llama.cpp"""
    log_message("开始下载llama.cpp CUDA版本...")
    
    # 下载到当前目录
    zip_file = os.path.join(PROJECT_ROOT, "llama_cpp_cuda.zip")
    
    # 尝试多个下载源
    for i, url in enumerate(LLAMA_CPP_CUDA_URLS, 1):
        log_message(f"尝试下载源 {i}/{len(LLAMA_CPP_CUDA_URLS)}")
        log_message(f"下载链接: {url}")
        log_message(f"正在下载到: {zip_file}")
        
        if download_file(url, zip_file):
            # 下载成功，跳出循环
            break
        else:
            # 下载失败，尝试下一个源
            log_message(f"下载源 {i} 失败，尝试下一个源...")
            # 删除可能损坏的文件
            if os.path.exists(zip_file):
                os.remove(zip_file)
            continue
    else:
        # 所有源都失败
        log_message("所有下载源都失败，退出")
        return False
    
    # 检查文件是否存在
    if not os.path.exists(zip_file):
        log_message("下载文件不存在")
        return False
    
    # 检查文件大小
    file_size = os.path.getsize(zip_file)
    log_message(f"下载文件大小: {file_size / (1024 * 1024):.2f} MB")
    
    if file_size < 10 * 1024 * 1024:  # 小于10MB，可能下载失败
        log_message("警告: 下载文件可能不完整")
    
    # 创建llama_cpp目录
    if not os.path.exists(LLAMA_CPP_DIR):
        os.makedirs(LLAMA_CPP_DIR, exist_ok=True)
        log_message(f"创建目录: {LLAMA_CPP_DIR}")
    else:
        log_message(f"目录已存在: {LLAMA_CPP_DIR}")
        # 清空目录
        for item in os.listdir(LLAMA_CPP_DIR):
            item_path = os.path.join(LLAMA_CPP_DIR, item)
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
        log_message("清空目录内容")
    
    # 解压文件
    log_message("正在解压...")
    if not extract_zip(zip_file, LLAMA_CPP_DIR):
        log_message("解压失败，退出")
        return False
    
    # 清理下载的zip文件
    try:
        os.remove(zip_file)
        log_message(f"已删除临时下载文件: {zip_file}")
    except Exception as e:
        log_message(f"删除临时文件失败: {str(e)}")
    
    # 验证解压结果
    if not os.listdir(LLAMA_CPP_DIR):
        log_message("解压后目录为空")
        return False
    
    # 检查是否有llama-cli可执行文件
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
