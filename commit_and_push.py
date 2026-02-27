import subprocess
import os

def run_command(cmd):
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        print(f"命令: {cmd}")
        print(f"输出: {result.stdout}")
        if result.stderr:
            print(f"错误: {result.stderr}")
        return result.returncode == 0
    except Exception as e:
        print(f"执行命令出错: {e}")
        return False

print("=== 开始提交并推送修改 ===")

# 1. 提交代码
print("\n1. 创建提交...")
success = run_command('git commit -m "Add FFmpeg download script"')

if success:
    print("\n2. 推送到 GitHub...")
    success = run_command('git push')
    
    if success:
        print("\n=== 上传成功！ ===")
    else:
        print("\n=== 推送失败 ===")
else:
    print("\n=== 提交失败，可能没有需要提交的文件 ===")
