import os
import cv2
import numpy as np

# Disjoint Set Union (并查集) 用於處理等價關係與傳遞閉包 (Transitive Closure)
class DisjointSet:
    def __init__(self, n=100000):
        self.parent = list(range(n))

    def find(self, i):
        path = []
        while self.parent[i] != i:
            path.append(i)
            i = self.parent[i]
        for node in path:
            self.parent[node] = i
        return i

    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            if root_i < root_j:
                self.parent[root_j] = root_i
            else:
                self.parent[root_i] = root_j

def otsu_threshold(gray_img):
    total_pixels = gray_img.size
    hist = np.zeros(256, dtype=int)
    for p in gray_img.ravel():
        hist[p] += 1

    min_within_class_var = float('inf')
    best_threshold = 0

    for t in range(256):
        n1 = np.sum(hist[:t + 1])
        n2 = np.sum(hist[t + 1:])
        if n1 == 0 or n2 == 0:
            continue

        intensities_1 = np.arange(0, t + 1)
        mean_1 = np.sum(intensities_1 * hist[:t + 1]) / n1
        var_1 = np.sum(((intensities_1 - mean_1) ** 2) * hist[:t + 1]) / n1

        intensities_2 = np.arange(t + 1, 256)
        mean_2 = np.sum(intensities_2 * hist[t + 1:]) / n2
        var_2 = np.sum(((intensities_2 - mean_2) ** 2) * hist[t + 1:]) / n2

        within_class_var = (n1 / total_pixels) * var_1 + (n2 / total_pixels) * var_2
        if within_class_var < min_within_class_var:
            min_within_class_var = within_class_var
            best_threshold = t

    return best_threshold

def two_pass_connected_components(binary_img):
    """
    Two-Pass Algorithm (使用 4-connectivity)
    binary_img: 2D array, 255 代表目標前景, 0 代表背景
    """
    h, w = binary_img.shape
    labels = np.zeros((h, w), dtype=int)
    dsu = DisjointSet(h * w + 1)
    next_label = 1

    # Pass 1: Label assignment, propagation, and equivalence recording
    for y in range(h):
        for x in range(w):
            if binary_img[y, x] == 0:
                continue

            neighbors = []
            # 上方鄰居
            if y > 0 and labels[y - 1, x] > 0:
                neighbors.append(labels[y - 1, x])
            # 左方鄰居
            if x > 0 and labels[y, x - 1] > 0:
                neighbors.append(labels[y, x - 1])

            if len(neighbors) == 0:
                labels[y, x] = next_label
                next_label += 1
            else:
                min_label = min(neighbors)
                labels[y, x] = min_label
                for neighbor in neighbors:
                    dsu.union(min_label, neighbor)

    # Pass 2: Label translation (替換為根標籤並重新編號)
    resolved_map = {}
    canonical_id = 1
    output_labels = np.zeros((h, w), dtype=int)

    for y in range(h):
        for x in range(w):
            if labels[y, x] > 0:
                root = dsu.find(labels[y, x])
                if root not in resolved_map:
                    resolved_map[root] = canonical_id
                    canonical_id += 1
                output_labels[y, x] = resolved_map[root]

    return output_labels, canonical_id - 1

def main():
    os.makedirs("./output", exist_ok=True)
    img_path = "./data/connected_component.jpg"
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"找不到圖片：{img_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th = otsu_threshold(gray)

    # 二值化處理
    binary = np.zeros_like(gray)
    binary[gray > th] = 255
    cv2.imwrite("./output/1_otsu.jpg", binary)
    print(f"Otsu 二值化完成 (閾值: {th})，儲存至 ./output/1_otsu.jpg")

    # Two-Pass 尋找連通元
    labeled_img, num_components = two_pass_connected_components(binary)
    print(f"找到 {num_components} 個連通元件")

    # 為每個 component 著上不同隨機色彩 (背景 label=0 維持純黑)
    np.random.seed(42)
    colors = np.random.randint(50, 256, size=(num_components + 1, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]

    colored_output = colors[labeled_img]
    cv2.imwrite("./output/1_connected_component.jpg", colored_output)
    print("已儲存彩色連通元結果至 ./output/1_connected_component.jpg")

if __name__ == "__main__":
    main()