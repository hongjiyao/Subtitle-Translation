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

print("=== 开始强制提交 ===")

# 1. 添加所有更改（包括修改和删除的文件）
print("\n1. 添加所有更改...")
run_command('git add -A')

# 2. 强制提交
print("\n2. 强制提交...")
run_command('git commit -m "Force commit: update all files" --force')

# 3. 强制推送到 GitHub
print("\n3. 强制推送到 GitHub...")
success = run_command('git push -f origin main')

if success:
    print("\n=== 强制提交成功！ ===")
else:
    print("\n=== 强制提交失败 ===")
