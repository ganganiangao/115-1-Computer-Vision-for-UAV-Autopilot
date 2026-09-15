import subprocess
import sys
from pathlib import Path

# 欲執行的腳本列表
scripts = ["q1.py", "q2.py", "q3.py"]

for script in scripts:
    script_path = Path(script)
    if not script_path.exists():
        print(f"[Error] 找不到檔案: {script}")
        sys.exit(1)

    print(f"=== 開始執行 {script} ===")
    # sys.executable 確保使用與當前環境完全相同的 python 直譯器
    result = subprocess.run([sys.executable, str(script_path)])
    
    if result.returncode != 0:
        print(f"[Failed] {script} 執行失敗，終止後續任務。")
        sys.exit(result.returncode)
    print("")

print("所有程式執行完成！輸出檔案請至 ./output 查看。")