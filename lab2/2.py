import os
import cv2
import numpy as np

def otsu_threshold(gray_img):
    total_pixels = gray_img.size

    # 1. 計算影像直方圖與每個灰階強度的機率 P(i)
    hist = np.zeros(256, dtype=int)
    for p in gray_img.ravel():
        hist[p] += 1
    prob = hist / total_pixels

    min_within_class_var = float('inf')
    best_threshold = 0

    # 2. 依序將 0~255 當作閾值計算組內變異數和
    for t in range(256):
        # 劃分兩組：背景 (強度 <= t) 與 前景 (強度 > t)
        q1 = np.sum(prob[:t + 1])      # 背景像素權重和 灰階強度在 0 ~ t 的像素權重和
        q2 = np.sum(prob[t + 1:])      # 前景像素權重和 灰階強度在 t+1 ~ 255 的像素權重和

        # 若其中一組為空，跳過此閾值
        if q1 == 0 or q2 == 0:
            continue

        # 計算背景的平均值與變異數
        intensities_1 = np.arange(0, t + 1)  # [0, 1, 2, ..., t]
        mean_1 = np.sum(intensities_1 * prob[:t + 1]) / q1
        var_1 = np.sum(((intensities_1 - mean_1) ** 2) * prob[:t + 1]) / q1

        # 計算前景的平均值與變異數
        intensities_2 = np.arange(t + 1, 256)  # [t+1, t+2, ..., 255]
        mean_2 = np.sum(intensities_2 * prob[t + 1:]) / q2
        var_2 = np.sum(((intensities_2 - mean_2) ** 2) * prob[t + 1:]) / q2

        # 組內變異數總和 (Within-class variance)
        within_class_var = q1 * var_1 + q2 * var_2

        # 尋找總和最小的閾值
        if within_class_var < min_within_class_var:
            min_within_class_var = within_class_var
            best_threshold = t

    return best_threshold

def main():
    os.makedirs("./output", exist_ok=True)
    img_path = "./data/otsu.jpg"
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"找不到圖片：{img_path}")

    # 轉為灰階圖
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 尋找最佳閾值
    best_t = otsu_threshold(gray)
    print(f"Otsu 計算的最佳閾值 (Threshold): {best_t}")

    # 二值化處理：大於閾值設為 255，其餘設為 0
    binary_img = np.zeros_like(gray)
    binary_img[gray > best_t] = 255

    cv2.imwrite("./output/2_result.jpg", binary_img)
    print("已儲存 2 二值化結果至 ./output/2_result.jpg")

if __name__ == "__main__":
    main()