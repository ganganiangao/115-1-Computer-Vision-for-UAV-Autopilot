import cv2
import numpy as np

# ----------------- 步驟 1: 載入看板影像與選定 4 點 -----------------
billboard_path = "caption.jpg"  # 請替換為助教提供的看板圖片路徑
bg_img = cv2.imread(billboard_path)

if bg_img is None:
    # 若無圖片，建立一張黑色預設看板供測試
    print(f"未找到 {billboard_path}，使用測試底圖示範。")
    bg_img = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.putText(bg_img, "Billboard Area", (500, 360), cv2.FONT_HERSHEY_SIMPLEX, 2, (100, 100, 100), 3)

clicked_pts = []

def on_mouse(event, x, y, flags, param):
    # 畫點，少於 4 個就繼續監聽滑鼠左鍵按的位置
    if event == cv2.EVENT_LBUTTONDOWN and len(clicked_pts) < 4:
        clicked_pts.append([x, y])  # 加入左鍵點的 [x, y]
        cv2.circle(param, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(param, str(len(clicked_pts)), (x + 8, y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Select 4 Corners", param)

clone_img = bg_img.copy()  # 複製一張背景來點四個點
cv2.imshow("Select 4 Corners", clone_img)
cv2.setMouseCallback("Select 4 Corners", on_mouse, clone_img)  # 將滑鼠事件與名為 "Select 4 Corners" 的視窗綁定

print("請依序點選看板的 4 個角：1.左上 -> 2.右上 -> 3.右下 -> 4.左下")
while len(clicked_pts) < 4:
    key = cv2.waitKey(20) & 0xFF
    if key == 27:
        break
cv2.destroyAllWindows()

if len(clicked_pts) < 4:
    # 備用預設座標 (左上, 右上, 右下, 左下)
    print("未選滿 4 點，採用預設目標區域。")
    dst_pts = np.float32([[594, 148], [886, 58], [1015, 696], [614, 731]])
else:
    dst_pts = np.float32(clicked_pts)

# ----------------- 步驟 2: 預先計算映射座標與插值權重 -----------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("無法開啟攝影機！")
    exit()

ret, sample_frame = cap.read()
if not ret:
    print("無法取得 Webcam 畫面！")
    exit()

h_src, w_src = sample_frame.shape[:2]
# 來源 Webcam 影像四個角落 (對齊順序：左上, 右上, 右下, 左下)
# 取得攝影機畫面本身的四個頂點座標
src_pts = np.float32([
    [0, 0],
    [w_src - 1, 0],
    [w_src - 1, h_src - 1],
    [0, h_src - 1]
])

# 透過目標多邊形建立遮罩，鎖定看板區域內的像素點
h_bg, w_bg = bg_img.shape[:2]  
mask = np.zeros((h_bg, w_bg), dtype=np.uint8)
cv2.fillPoly(mask, [dst_pts.astype(np.int32)], 255)  # 在全黑畫布上，將使用者選取的 4 點多邊形塗成白色
dst_y, dst_x = np.where(mask == 255)  # 直接抓出所有白色區域的像素座標 (dst_x, dst_y)

# 使用 cv2.getPerspectiveTransform 取得反向轉換矩陣 (看板 -> Webcam)
# 反向映射能避免向前映射產生的空隙與破洞
M_inv = cv2.getPerspectiveTransform(dst_pts, src_pts)

# 將看板上的目標像素座標轉為齊次座標 [x, y, 1]^T 進行投影
dst_homo = np.stack([dst_x, dst_y, np.ones_like(dst_x)], axis=0)
src_homo = M_inv @ dst_homo

# 得到對應至來源 Webcam 上的浮點數座標 (u, v)
u = src_homo[0] / src_homo[2]
v = src_homo[1] / src_homo[2]

# 過濾超出來源影像邊界的安全範圍
valid = (u >= 0) & (u < w_src - 1) & (v >= 0) & (v < h_src - 1)
u_val = u[valid]
v_val = v[valid]
out_x = dst_x[valid]
out_y = dst_y[valid]

# 雙線性插值的 4 個鄰近整數像素索引
x1 = np.floor(u_val).astype(np.int32)
y1 = np.floor(v_val).astype(np.int32)
x2 = x1 + 1
y2 = y1 + 1

# 計算距離權重
dx = (u_val - x1)[:, None]
dy = (v_val - y1)[:, None]

w11 = (1.0 - dx) * (1.0 - dy)
w21 = dx * (1.0 - dy)
w12 = (1.0 - dx) * dy
w22 = dx * dy

print("開始即時 Warping 輸出，按 'ESC' 退出...")

# ----------------- 步驟 3: 即時影像雙線性插值串流 -----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 透過矩陣索引直接取出 4 鄰域像素值 (Vectorized Bilinear Interpolation)
    q11 = frame[y1, x1]
    q21 = frame[y1, x2]
    q12 = frame[y2, x1]
    q22 = frame[y2, x2]

    # 插值計算出目標像素顏色
    interp_pixels = q11 * w11 + q21 * w21 + q12 * w12 + q22 * w22

    # 貼回背景底圖
    result = bg_img.copy()
    result[out_y, out_x] = interp_pixels.astype(np.uint8)

    cv2.imshow("Task 2 - Real-time Warping Practice", result)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

