#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aria2文件下载工具
支持多线程下载、断点续传、进度显示等功能
"""

import os
import sys
import subprocess
import time
import zipfile

from utils.logger import get_optimal_threads, cleanup_duplicate_files
from utils.download_utils import find_aria2c, check_aria2_available, download_with_progress, verify_zip

class Aria2Downloader:
    """aria2下载器类"""
    
    def __init__(self, aria2c_path=None):
        """初始化下载器
        
        Args:
            aria2c_path: aria2c.exe的路径，如果为None则自动搜索
        """
        self.aria2c_path = aria2c_path or find_aria2c()
        self.optimal_threads = get_optimal_threads()
        
    def download_aria2(self, output_dir=None):
        """下载并解压aria2
        
        Args:
            output_dir: 输出目录，默认当前目录
            
        Returns:
            (success, message): (是否成功, 消息)
        """
        output_dir = output_dir or os.getcwd()
        aria2_url = "https://github.com/aria2/aria2/releases/download/release-1.37.0/aria2-1.37.0-win-64bit-build1.zip"
        backup_url = "https://ghproxy.com/https://github.com/aria2/aria2/releases/download/release-1.37.0/aria2-1.37.0-win-64bit-build1.zip"
        zip_path = os.path.join(output_dir, "aria2-1.37.0-win-64bit-build1.zip")
        
        print("开始下载aria2...")
        print(f"下载链接: {aria2_url}")
        print(f"保存路径: {zip_path}")
        print("=" * 80)
        
        max_attempts = 3
        for attempt in range(max_attempts):
            current_url = aria2_url if attempt == 0 else backup_url
            if attempt > 0:
                print(f"\n尝试备用链接...")
                print(f"备用链接: {current_url}")
            
            try:
                if download_with_progress(current_url, zip_path):
                    if not verify_zip(zip_path):
                        print("ZIP file is corrupted")
                        if os.path.exists(zip_path):
                            os.remove(zip_path)
                        continue
                    
                    print("\n解压aria2...")
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(output_dir)
                    
                    os.remove(zip_path)
                    
                    aria2c_path = os.path.join(output_dir, "aria2-1.37.0-win-64bit-build1", "aria2c.exe")
                    if os.path.exists(aria2c_path):
                        self.aria2c_path = aria2c_path
                        return True, f"aria2下载成功: {aria2c_path}"
                    else:
                        return False, "aria2解压成功但未找到aria2c.exe"
                else:
                    print(f"下载失败，尝试第 {attempt + 1}/{max_attempts} 次...")
                    time.sleep(120)
                    continue
            except Exception as e:
                print(f"下载出错: {str(e)}")
                print(f"尝试第 {attempt + 1}/{max_attempts} 次...")
                time.sleep(120)
                continue
        
        return False, "下载aria2失败，请检查网络连接"


    
    def check_aria2(self):
        """检查aria2是否可用"""
        self.aria2c_path = find_aria2c()
        return check_aria2_available(self.aria2c_path)
    
    def download(self, url, output_path, threads=None, split=None, 
                max_connection_per_server=16, min_split_size="1M",
                file_allocation="none", timeout=3600):
        """下载文件
        
        Args:
            url: 下载链接
            output_path: 输出文件路径
            threads: 线程数，默认使用最佳线程数
            split: 分片数，默认与线程数相同
            max_connection_per_server: 每服务器最大连接数
            min_split_size: 最小分片大小
            file_allocation: 文件分配方式
            timeout: 超时时间（秒）
            
        Returns:
            (success, message): (是否成功, 消息)
        """
        # 检查aria2是否可用
        is_available, msg = self.check_aria2()
        if not is_available:
            return False, f"aria2不可用: {msg}"
        
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # 检查文件是否已存在
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return True, f"文件已存在，跳过下载: {os.path.basename(output_path)}"
        
        # 设置参数
        threads = threads or self.optimal_threads
        split = split or threads
        
        # 构建命令
        cmd = [
            self.aria2c_path,
            "-x", str(threads),           # 最大连接数
            "-s", str(split),           # 分片数
            "-k", "1M",           # 分片大小
            "--max-connection-per-server", str(max_connection_per_server),
            "--split", str(split),
            "--min-split-size", min_split_size,
            "--file-allocation", file_allocation,
            "-o", os.path.basename(output_path),
            "-d", output_dir,
            "--summary-interval", "1",  # 每秒输出一次进度
            url
        ]
        
        print(f"开始下载: {os.path.basename(output_path)}")
        print(f"下载链接: {url}")
        print(f"输出路径: {output_path}")
        print(f"线程数: {threads}")
        print(f"分片数: {split}")
        print("=" * 80)
        
        try:
            # 执行命令
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            
            if result.returncode == 0:
                # 清理可能的重复文件
                cleanup_duplicate_files(output_dir)
                return True, f"下载成功: {os.path.basename(output_path)}"
            else:
                error_msg = result.stderr or "未知错误"
                return False, f"下载失败: {error_msg[:200]}"
                
        except subprocess.TimeoutExpired:
            return False, f"下载超时（{timeout}秒）"
        except Exception as e:
            return False, f"下载出错: {str(e)}"
    
    def download_multiple(self, downloads, threads=None):
        """批量下载多个文件
        
        Args:
            downloads: 下载列表，每个元素为 (url, output_path) 元组
            threads: 线程数，默认使用最佳线程数
            
        Returns:
            (success_count, failed_count, failed_downloads): (成功数, 失败数, 失败的下载)
        """
        success_count = 0
        failed_count = 0
        failed_downloads = []
        
        for i, (url, output_path) in enumerate(downloads, 1):
            print(f"\n[{i}/{len(downloads)}] 处理: {os.path.basename(output_path)}")
            success, msg = self.download(url, output_path, threads=threads)
            
            if success:
                print(f"✓ {msg}")
                success_count += 1
            else:
                print(f"✗ {msg}")
                failed_count += 1
                failed_downloads.append((url, output_path, msg))
            
        return success_count, failed_count, failed_downloads
    
    def get_version(self):
        """获取aria2版本"""
        is_available, msg = self.check_aria2()
        if is_available:
            return msg
        else:
            return None

def main():
    """主函数"""
    print("aria2文件下载工具")
    print("=" * 80)
    
    # 创建下载器实例
    downloader = Aria2Downloader()
    
    # 检查aria2是否可用
    is_available, msg = downloader.check_aria2()
    if not is_available:
        print(f"aria2不可用: {msg}")
        print("尝试下载aria2...")
        
        # 下载aria2
        download_success, download_msg = downloader.download_aria2()
        if not download_success:
            print(f"错误: {download_msg}")
            return 1
        
        # 重新检查aria2是否可用
        is_available, msg = downloader.check_aria2()
        if not is_available:
            print(f"错误: {msg}")
            return 1
    
    print(f"aria2版本: {msg}")
    print(f"最佳线程数: {downloader.optimal_threads}")
    print()
    
    # 示例下载
    test_downloads = [
        # 示例1：下载一个小文件
        ("https://example.com/test.txt", "downloads/test.txt"),
        # 示例2：下载一个较大的文件
        # ("https://example.com/large_file.zip", "downloads/large_file.zip"),
    ]
    
    print("开始测试下载...")
    success_count, failed_count, failed_downloads = downloader.download_multiple(test_downloads)
    
    print("\n" + "=" * 80)
    print("下载完成")
    print(f"成功: {success_count}")
    print(f"失败: {failed_count}")
    
    if failed_downloads:
        print("\n失败的下载:")
        for url, output_path, msg in failed_downloads:
            print(f"- {os.path.basename(output_path)}: {msg}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
