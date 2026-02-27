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

# 1. 添加所有更改
print("\n1. 添加更改...")
run_command('git add README.md push_changes.py')

# 2. 提交
print("\n2. 创建提交...")
run_command('git commit -m "Update README and push scripts"')

# 3. 强制推送到 GitHub（覆盖远程）
print("\n3. 推送到 GitHub...")
success = run_command('git push -f origin main')

if success:
    print("\n=== 上传成功！ ===")
else:
    print("\n=== 推送失败 ===")
