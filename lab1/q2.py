import os
import cv2
import numpy as np


def bilinear_interpolation(img, scale=3):

    height, width = img.shape[:2]
    new_height = int(height * scale)
    new_width = int(width * scale)

    is_color = (len(img.shape) == 3)

    # 產生新影像每個像素對應回原圖的座標 (浮點數)
    # np.arange 產生 0 ~ new_height-1 / new_width-1 的整數座標，再除以 scale
    ny = np.arange(new_height)
    nx = np.arange(new_width)

    orig_y = ny / scale
    orig_x = nx / scale

    # 取floor得到左上角座標 x1, y1
    x1 = np.floor(orig_x).astype(np.int32)
    y1 = np.floor(orig_y).astype(np.int32)

    # x2, y2 需 clip 到邊界內，避免 index out of range
    x2 = np.clip(x1 + 1, 0, width - 1)
    y2 = np.clip(y1 + 1, 0, height - 1)
    x1 = np.clip(x1, 0, width - 1)
    y1 = np.clip(y1, 0, height - 1)

    # u, w 為小數部分權重
    u = (orig_x - np.floor(orig_x)).astype(np.float32)  # shape: (new_width,)
    w = (orig_y - np.floor(orig_y)).astype(np.float32)  # shape: (new_height,)

    # 用 meshgrid 展開成跟輸出影像一樣的 2D 網格，方便做逐元素運算
    X1, Y1 = np.meshgrid(x1, y1)  # shape: (new_height, new_width)
    X2, Y2 = np.meshgrid(x2, y2)
    U, W = np.meshgrid(u, w)

    img_f = img.astype(np.float32)

    if is_color:
        # 用 fancy indexing 一次取出所有點的四個鄰居 (shape: new_height, new_width, channels)
        Q11 = img_f[Y1, X1]
        Q21 = img_f[Y1, X2]
        Q12 = img_f[Y2, X1]
        Q22 = img_f[Y2, X2]

        U = U[:, :, None]  # 擴展維度以便跟 channel 相乘
        W = W[:, :, None]
    else:
        Q11 = img_f[Y1, X1]
        Q21 = img_f[Y1, X2]
        Q12 = img_f[Y2, X1]
        Q22 = img_f[Y2, X2]

    new_img = (1 - U) * (1 - W) * Q11 + \
              U * (1 - W) * Q21 + \
              (1 - U) * W * Q12 + \
              U * W * Q22

    return np.clip(new_img, 0, 255).astype(np.uint8)


if __name__ == "__main__":
# 建立 output 目錄（若不存在）
    os.makedirs("./output", exist_ok=True)

    input_path = os.path.join("./data", "q2.jpg")
    img = cv2.imread(input_path)

    if img is None:
        print(f"Error: 找不到輸入檔案 {input_path}，請確認路徑。")
        exit()

    # 自行實作雙線性內插法 (放大 3 倍)
    result = bilinear_interpolation(img, scale=3)

    # 儲存結果至 ./output
    output_path = os.path.join("./output", "q2_result.jpg")
    cv2.imwrite(output_path, result)

    print(f"處理完成！已輸出至：{output_path}")