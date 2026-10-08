import os
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "output")

# ---------------- 參數 ----------------
MIN_AREA = 800        # 面積門檻 T（以原始影片解析度的像素數計）
PROC_WIDTH = 480      # 為了加速，connected component 在縮小後的遮罩上做（Python 迴圈較慢）
SHOW = False          # True 則即時顯示畫面（WSL 沒有 GUI 時請維持 False）
BOX_COLOR = (0, 255, 0)
BOX_WIDTH = 2


def connected_components_with_stats(binary):
    """
    Two-pass connected component（修改自第一題），同時計算每個區域的面積與外框。
    回傳 labels, areas, bboxes
      areas[k]  : 區域 k 的面積
      bboxes[k] : (x_min, y_min, x_max, y_max)
    """
    h, w = binary.shape
    labels = [[0] * (w + 2) for _ in range(h + 2)]
    parent = [0]

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

    # Pass 1
    ys, xs = np.nonzero(binary)
    for y, x in zip(ys.tolist(), xs.tolist()):
        py, px = y + 1, x + 1
        up = labels[py - 1]
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
                    union(m, l)

    # resolve
    lut = np.zeros(len(parent), dtype=np.int32)
    root_to_new = {}
    for i in range(1, len(parent)):
        r = find(i)
        if r not in root_to_new:
            root_to_new[r] = len(root_to_new) + 1
        lut[i] = root_to_new[r]
    n = len(root_to_new)

    # Pass 2：label translation，並同時統計面積與最外圍四個邊
    lab = np.array(labels, dtype=np.int32)[1:-1, 1:-1]
    final = lut[lab]

    fl = final[ys, xs]
    areas = np.bincount(fl, minlength=n + 1)

    x_min = np.full(n + 1, w, dtype=np.int32)
    y_min = np.full(n + 1, h, dtype=np.int32)
    x_max = np.full(n + 1, -1, dtype=np.int32)
    y_max = np.full(n + 1, -1, dtype=np.int32)
    np.minimum.at(x_min, fl, xs)
    np.minimum.at(y_min, fl, ys)
    np.maximum.at(x_max, fl, xs)
    np.maximum.at(y_max, fl, ys)
    bboxes = np.stack([x_min, y_min, x_max, y_max], axis=1)

    return final, areas, bboxes


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    video_path = os.path.join(DATA_DIR, "mot.mp4")
    out_path = os.path.join(OUT_DIR, "2_result.mp4")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"無法開啟影片：{video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 總 frame 數（部分影片可能為 0 或不準）
    print(f"影片有 {total} frames")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    scale = min(1.0, PROC_WIDTH / W)
    pw, ph = int(W * scale), int(H * scale)
    min_area_small = MIN_AREA * scale * scale

    backSub = cv2.createBackgroundSubtractorMOG2()
    shadow_val = backSub.getShadowValue()  # 預設 127
    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 背景相減 + 去除陰影
        fgmask = backSub.apply(frame)
        _, nmask = cv2.threshold(fgmask, shadow_val, 255, cv2.THRESH_BINARY)

        # 縮小、去雜訊（先 open 去掉小雜點，再 close 補起人體破洞）
        small = cv2.resize(nmask, (pw, ph), interpolation=cv2.INTER_NEAREST)
        small = cv2.morphologyEx(small, cv2.MORPH_OPEN, open_k)
        small = cv2.morphologyEx(small, cv2.MORPH_CLOSE, close_k)

        # Connected component（含面積），面積 > T 才畫框
        _, areas, bboxes = connected_components_with_stats(small)
        for k in range(1, len(areas)):
            if areas[k] > min_area_small:
                x1, y1, x2, y2 = bboxes[k]
                p1 = (int(x1 / scale), int(y1 / scale))
                p2 = (int((x2 + 1) / scale), int((y2 + 1) / scale))
                cv2.rectangle(frame, p1, p2, BOX_COLOR, BOX_WIDTH)

        writer.write(frame)
        if SHOW:
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        idx += 1
        if idx % 30 == 0:
            print(f"已處理 {idx} / {total}" if total > 0 else f"已處理 {idx}")

    cap.release()
    writer.release()
    if SHOW:
        cv2.destroyAllWindows()
    print(f"完成，已處理 {idx} / {total if total > 0 else idx}，已儲存至 {out_path}")


if __name__ == "__main__":
    main()