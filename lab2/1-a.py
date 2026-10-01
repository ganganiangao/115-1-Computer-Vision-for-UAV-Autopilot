import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def compute_histogram_and_lut(channel):
    """
    計算單通道影像的直方圖、CDF 並建立映射表 (Lookup Table)
    """
    # 1. 計算直方圖 (各強度 0~255 出現次數)
    hist = np.zeros(256, dtype=int)
    for pixel in channel.ravel():
        hist[pixel] += 1
        
    # 2. 計算累計分佈 (CDF / Acc Sum)
    total_pixels = channel.size
    cdf = np.zeros(256, dtype=float)
    current_sum = 0
    for i in range(256):
        current_sum += hist[i]
        cdf[i] = current_sum / total_pixels
        
    # 3. 映射到 0~255 並四捨五入取整數
    lut = np.round(cdf * 255).astype(np.uint8)
    
    return hist, lut

def equalize_channel(channel):
    hist, lut = compute_histogram_and_lut(channel)
    # 利用 LUT 映射產生新像素值
    equalized = lut[channel]
    return equalized, hist

def main():
    os.makedirs("./output", exist_ok=True)
    img_path = "./data/histogram.jpg"
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"找不到圖片：{img_path}")

    # 分離 B, G, R 通道
    b, g, r = cv2.split(img)

    # 分別做直方圖等化
    b_eq, b_hist_orig = equalize_channel(b)
    g_eq, g_hist_orig = equalize_channel(g)
    r_eq, r_hist_orig = equalize_channel(r)

    # 合併三個通道
    output_img = cv2.merge([b_eq, g_eq, r_eq])
    cv2.imwrite("./output/1-a_result.jpg", output_img)
    print("已儲存 1-a 影像結果至 ./output/1-a_result.jpg")

    # 繪製直方圖圖表 (比較等化前與等化後)
    b_hist_eq, _ = compute_histogram_and_lut(b_eq)
    g_hist_eq, _ = compute_histogram_and_lut(g_eq)
    r_hist_eq, _ = compute_histogram_and_lut(r_eq)

    plt.figure(figsize=(12, 6))

    # 原圖直方圖
    plt.subplot(1, 2, 1)
    plt.plot(b_hist_orig, color='blue', label='Blue Channel')
    plt.plot(g_hist_orig, color='green', label='Green Channel')
    plt.plot(r_hist_orig, color='red', label='Red Channel')
    plt.title('Original BGR Histogram')
    plt.xlabel('Pixel Intensity (0-255)')
    plt.ylabel('Pixel Count')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    # 等化後直方圖
    plt.subplot(1, 2, 2)
    plt.plot(b_hist_eq, color='blue', label='Blue Channel')
    plt.plot(g_hist_eq, color='green', label='Green Channel')
    plt.plot(r_hist_eq, color='red', label='Red Channel')
    plt.title('Equalized BGR Histogram')
    plt.xlabel('Pixel Intensity (0-255)')
    plt.ylabel('Pixel Count')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig("./output/1-a_histogram.png", dpi=300)
    plt.close()
    print("已儲存直方圖圖表至 ./output/1-a_histogram.png")

if __name__ == "__main__":
    main()