import os
import cv2
import numpy as np

def main():
    input_path = os.path.join("./data", "q3.jpg")
    output_dir = "./output"
    output_path = os.path.join(output_dir, "q3_result.jpg")

    # 確保輸出資料夾存在，若無則自動建立
    os.makedirs(output_dir, exist_ok=True)

    # 讀取測試圖片
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: 找不到圖片，請確認檔案是否存在於: {input_path}")
        return

    # 1. 轉為灰階
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. 高斯模糊 (去雜訊)
    blur = cv2.GaussianBlur(gray, (5, 5), 1.0)

    # 3. 定義 Sobel 卷積核 (Kernel)
    kernel_x = np.array([
        [ 1,  0, -1],
        [ 2,  0, -2],
        [ 1,  0, -1]
    ], dtype=np.float32)

    kernel_y = np.array([
        [ 1,  2,  1],
        [ 0,  0,  0],
        [-1, -2, -1]
    ], dtype=np.float32)

    # 使用 cv2.filter2D 進行卷積計算 Gx 與 Gy
    # 注意：ddepth 設為 cv2.CV_64F 以避免負值截斷
    gx = cv2.filter2D(blur, cv2.CV_64F, kernel_x)
    gy = cv2.filter2D(blur, cv2.CV_64F, kernel_y)

    # 4. 計算邊緣強度 G = sqrt(Gx^2 + Gy^2)
    magnitude = np.sqrt(gx**2 + gy**2)

    # 5. 限制像素範圍於 0~255 並轉為 uint8
    edge_img = np.clip(magnitude, 0, 255).astype(np.uint8)

    # 儲存結果到 ./output 資料夾
    cv2.imwrite(output_path, edge_img)
    print(f"已成功輸出結果至: {output_path}")

if __name__ == "__main__":
    main()