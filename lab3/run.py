import subprocess
import sys
import os

scripts = [
    "1.py",
    "2.py"
]

def run_script(script_name):
    if not os.path.exists(script_name):
        print(f"[錯誤] 找不到檔案: {script_name}")
        return False

    print(f"\n{'='*20} 開始執行 {script_name} {'='*20}")
    result = subprocess.run([sys.executable, script_name])

    if result.returncode == 0:
        print(f"[成功] {script_name} 執行完成！")
        return True
    else:
        print(f"[失敗] {script_name} 執行中斷，錯誤代碼: {result.returncode}")
        return False

def main():
    os.makedirs("./output", exist_ok=True)

    for script in scripts:
        success = run_script(script)
        if not success:
            print(f"\n執行過程中斷，請檢查 {script} 的錯誤訊息。")
            sys.exit(1)

    print(f"\n{'='*50}")
    print("所有作業程式執行完畢！所有輸出結果已存放在 ./output/")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
