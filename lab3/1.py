import os
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "output")


# ---------------------------------------------------------------
# Otsu (上次作業)
# ---------------------------------------------------------------
def otsu_threshold(gray_img):
    total_pixels = gray_img.size
    hist = np.bincount(gray_img.ravel(), minlength=256)
    prob = hist / total_pixels

    min_within_class_var = float('inf')
    best_threshold = 0

    for t in range(256):
        q1 = np.sum(prob[:t + 1])
        q2 = np.sum(prob[t + 1:])
        if q1 == 0 or q2 == 0:
            continue

        intensities_1 = np.arange(0, t + 1)
        mean_1 = np.sum(intensities_1 * prob[:t + 1]) / q1
        var_1 = np.sum(((intensities_1 - mean_1) ** 2) * prob[:t + 1]) / q1

        intensities_2 = np.arange(t + 1, 256)
        mean_2 = np.sum(intensities_2 * prob[t + 1:]) / q2
        var_2 = np.sum(((intensities_2 - mean_2) ** 2) * prob[t + 1:]) / q2

        within_class_var = q1 * var_1 + q2 * var_2
        if within_class_var < min_within_class_var:
            min_within_class_var = within_class_var
            best_threshold = t

    return best_threshold


# ---------------------------------------------------------------
# Two-Pass Connected Component (8-connectivity)
# ---------------------------------------------------------------
def connected_components(binary):
    """
    binary: 2D array，非 0 為前景
    回傳 (labels, num_labels)；背景為 0，區域編號 1 ~ num_labels
    """
    h, w = binary.shape

    # 外面包一圈 0，避免處理邊界
    labels = [[0] * (w + 2) for _ in range(h + 2)]
    parent = [0]  # union-find，parent[0] 代表背景

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]  # path compression
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

    # ---- Pass 1: label assignment / propagation / equivalence ----
    ys, xs = np.nonzero(binary)  # row-major 順序，等同由左到右、由上到下掃描
    for y, x in zip(ys.tolist(), xs.tolist()):
        py, px = y + 1, x + 1
        up = labels[py - 1]
        # 已掃描過的鄰居：W, NW, N, NE
        neigh = [l for l in (labels[py][px - 1], up[px - 1], up[px], up[px + 1]) if l]

        if not neigh:
            new_label = len(parent)
            parent.append(new_label)
            labels[py][px] = new_label
        else:
            m = min(neigh)
            labels[py][px] = m
            for l in neigh:
                if l != m:
                    union(m, l)  # 兩個不同 label 傳到同一像素 -> 等價關係

    # ---- resolve: 找出等價關係的 transitive closure，並重新編成連續編號 ----
    lut = np.zeros(len(parent), dtype=np.int32)
    root_to_new = {}
    for i in range(1, len(parent)):
        r = find(i)
        if r not in root_to_new:
            root_to_new[r] = len(root_to_new) + 1
        lut[i] = root_to_new[r]

    # ---- Pass 2: label translation ----
    lab = np.array(labels, dtype=np.int32)[1:-1, 1:-1]
    final = lut[lab]
    return final, len(root_to_new)


def colorize(labels, num_labels):
    rng = np.random.default_rng(42)
    colors = rng.integers(60, 256, size=(num_labels + 1, 3), dtype=np.uint8)
    colors[0] = (0, 0, 0)  # 背景維持黑色
    return colors[labels]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    img_path = os.path.join(DATA_DIR, "q1.png")
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"找不到圖片：{img_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Otsu 二值化
    best_t = otsu_threshold(gray)
    print(f"Otsu 最佳閾值: {best_t}")
    binary = np.zeros_like(gray)
    binary[gray > best_t] = 255
    cv2.imwrite(os.path.join(OUT_DIR, "1_otsu.jpg"), binary)

    # Connected component
    labels, num = connected_components(binary)
    print(f"Connected component 數量: {num}")
    result = colorize(labels, num)
    cv2.imwrite(os.path.join(OUT_DIR, "1_connected_component.jpg"), result)
    print("已儲存至 output/1_otsu.jpg 與 output/1_connected_component.jpg")


if __name__ == "__main__":
    main()