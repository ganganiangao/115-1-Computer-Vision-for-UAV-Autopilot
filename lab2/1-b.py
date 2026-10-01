import os
import cv2
import numpy as np

def equalize_channel(channel):
    # 計算直方圖
    hist = np.zeros(256, dtype=int)
    for pixel in channel.ravel():
        hist[pixel] += 1
        
    # 計算累計分佈 (CDF)
    total_pixels = channel.size
    cdf = np.zeros(256, dtype=float)
    current_sum = 0
    for i in range(256):
        current_sum += hist[i]
        cdf[i] = current_sum / total_pixels
        
    # 映射表
    lut = np.round(cdf * 255).astype(np.uint8)
    return lut[channel]

def main():
    os.makedirs("./output", exist_ok=True)
    img_path = "./data/histogram.jpg"
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"找不到圖片：{img_path}")

    # BGR 轉 HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # 僅對 V 通道 (明度) 做直方圖等化
    v_eq = equalize_channel(v)

    # 合併回 HSV 並轉換回 BGR
    hsv_eq = cv2.merge([h, s, v_eq])
    output_img = cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)

    cv2.imwrite("./output/1-b_result.jpg", output_img)
    print("已儲存 1-b 影像結果至 ./output/1-b_result.jpg")

if __name__ == "__main__":
    main()