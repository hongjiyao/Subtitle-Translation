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
from tqdm import tqdm

# 强制使用 UTF-8 编码
os.environ['PYTHONIOENCODING'] = 'utf-8'

class Aria2Downloader:
    """aria2下载器类"""
    
    def __init__(self, aria2c_path=None):
        """初始化下载器
        
        Args:
            aria2c_path: aria2c.exe的路径，如果为None则自动搜索
        """
        self.aria2c_path = aria2c_path or self._find_aria2c()
        self.optimal_threads = self._get_optimal_threads()
        
    def _find_aria2c(self):
        """自动搜索aria2c.exe路径"""
        # 尝试多个可能的位置
        aria2c_paths = [
            # 当前目录
            os.path.join(os.getcwd(), "aria2c.exe"),
            # 直接在aria2目录
            os.path.join(os.getcwd(), "aria2c.exe"),
            # 常见的aria2目录结构
            os.path.join(os.getcwd(), "aria2-1.37.0-win-64bit-build1", "aria2c.exe"),
            os.path.join(os.getcwd(), "aria2", "aria2c.exe"),
            os.path.join(os.getcwd(), "aria2", "aria2-1.37.0-win-64bit-build1", "aria2c.exe"),
            # 上级目录
            os.path.join(os.path.dirname(os.getcwd()), "aria2c.exe"),
            os.path.join(os.path.dirname(os.getcwd()), "aria2-1.37.0-win-64bit-build1", "aria2c.exe"),
            os.path.join(os.path.dirname(os.getcwd()), "aria2", "aria2c.exe"),
            os.path.join(os.path.dirname(os.getcwd()), "aria2", "aria2-1.37.0-win-64bit-build1", "aria2c.exe"),
        ]
        
        # 遍历搜索所有可能的路径
        for path in aria2c_paths:
            if os.path.exists(path):
                return path
        
        # 搜索当前目录及其子目录
        for root, dirs, files in os.walk(os.getcwd()):
            if "aria2c.exe" in files:
                return os.path.join(root, "aria2c.exe")
        
        # 搜索上级目录及其子目录
        parent_dir = os.path.dirname(os.getcwd())
        if parent_dir != os.getcwd():
            for root, dirs, files in os.walk(parent_dir):
                if "aria2c.exe" in files:
                    return os.path.join(root, "aria2c.exe")
        
        return None
    
    def download_aria2(self, output_dir=None):
        """下载并解压aria2
        
        Args:
            output_dir: 输出目录，默认当前目录
            
        Returns:
            (success, message): (是否成功, 消息)
        """
        output_dir = output_dir or os.getcwd()
        # 主下载链接
        aria2_url = "https://github.com/aria2/aria2/releases/download/release-1.37.0/aria2-1.37.0-win-64bit-build1.zip"
        # 备用下载链接（GitHub镜像）
        backup_url = "https://ghproxy.com/https://github.com/aria2/aria2/releases/download/release-1.37.0/aria2-1.37.0-win-64bit-build1.zip"
        zip_path = os.path.join(output_dir, "aria2-1.37.0-win-64bit-build1.zip")
        
        print("开始下载aria2...")
        print(f"下载链接: {aria2_url}")
        print(f"保存路径: {zip_path}")
        print("=" * 80)
        
        # 尝试下载，支持重试和备用链接
        max_attempts = 3
        for attempt in range(max_attempts):
            # 选择下载链接
            current_url = aria2_url if attempt == 0 else backup_url
            if attempt > 0:
                print(f"\n尝试备用链接...")
                print(f"备用链接: {current_url}")
            
            # 下载zip文件
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
                        print(f"继续下载，已下载 {file_size / (1024*1024):.2f} MB")
                
                response = requests.get(current_url, stream=True, headers=resume_header, timeout=60)
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
                
                # 解压zip文件
                print("\n解压aria2...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(output_dir)
                
                # 删除zip文件
                os.remove(zip_path)
                
                # 验证aria2c.exe是否存在
                aria2c_path = os.path.join(output_dir, "aria2-1.37.0-win-64bit-build1", "aria2c.exe")
                if os.path.exists(aria2c_path):
                    self.aria2c_path = aria2c_path
                    return True, f"aria2下载成功: {aria2c_path}"
                else:
                    return False, "aria2解压成功但未找到aria2c.exe"
                    
            except Exception as e:
                print(f"下载出错: {str(e)}")
                print(f"尝试第 {attempt + 1}/{max_attempts} 次...")
                import time
                time.sleep(120)  # 等待120秒后重试
                # 不要删除部分下载的文件，以便断点续传
                continue
        
        return False, "下载aria2失败，请检查网络连接"


    
    def _get_optimal_threads(self):
        """根据系统配置获取最佳线程数"""
        try:
            import multiprocessing
            cpu_count = multiprocessing.cpu_count()
            optimal_threads = max(8, min(16, cpu_count * 2))
            return optimal_threads
        except Exception:
            return 16  # 默认使用16线程
    
    def check_aria2(self):
        """检查aria2是否可用（仅检查当前项目目录）"""
        # 重置aria2c_path，只在当前项目目录搜索
        self.aria2c_path = self._find_aria2c_in_current_project()
        
        if not self.aria2c_path:
            return False, "未找到aria2c.exe"
        
        if not os.path.exists(self.aria2c_path):
            return False, f"aria2c.exe不存在: {self.aria2c_path}"
        
        try:
            result = subprocess.run([self.aria2c_path, "--version"], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                version = result.stdout.split('\n')[0]
                return True, version
            else:
                return False, f"aria2c执行失败: {result.stderr}"
        except Exception as e:
            return False, f"检查aria2失败: {str(e)}"
    
    def _find_aria2c_in_current_project(self):
        """只在当前项目目录搜索aria2c.exe"""
        # 尝试多个可能的位置，但只在当前项目目录
        aria2c_paths = [
            # 当前目录
            os.path.join(os.getcwd(), "aria2c.exe"),
            # 常见的aria2目录结构
            os.path.join(os.getcwd(), "aria2-1.37.0-win-64bit-build1", "aria2c.exe"),
            os.path.join(os.getcwd(), "aria2", "aria2c.exe"),
            os.path.join(os.getcwd(), "aria2", "aria2-1.37.0-win-64bit-build1", "aria2c.exe"),
        ]
        
        # 遍历搜索所有可能的路径
        for path in aria2c_paths:
            if os.path.exists(path):
                return path
        
        # 搜索当前目录及其子目录
        for root, dirs, files in os.walk(os.getcwd()):
            if "aria2c.exe" in files:
                return os.path.join(root, "aria2c.exe")
        
        return None
    
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
                self._cleanup_duplicate_files(output_dir)
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
    
    def _cleanup_duplicate_files(self, directory):
        """清理目录中的重复文件（带有数字后缀的文件）"""
        if not os.path.exists(directory):
            return
        
        import re
        file_groups = {}
        
        for file in os.listdir(directory):
            match = re.match(r'^(.*)\.(\d+)\.(.*)$', file)
            if match:
                base_name = match.group(1)
                ext = match.group(3)
                original_file = f"{base_name}.{ext}"
                if original_file not in file_groups:
                    file_groups[original_file] = []
                file_groups[original_file].append(file)
        
        for original_file, duplicate_files in file_groups.items():
            original_path = os.path.join(directory, original_file)
            if os.path.exists(original_path):
                for duplicate_file in duplicate_files:
                    duplicate_path = os.path.join(directory, duplicate_file)
                    try:
                        os.remove(duplicate_path)
                    except Exception:
                        pass
    
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
