import os
import cv2
import numpy as np

# 建立 output 資料夾（若不存在則建立）
os.makedirs("./output", exist_ok=True)

# 讀取輸入圖片
input_path = os.path.join("./data", "q1.jpg")
img = cv2.imread(input_path)

if img is None:
    print(f"Error: 找不到輸入檔案 {input_path}，請確認路徑。")
    exit()

img_float = img.astype(np.float32)
B, G, R = cv2.split(img_float)

# =======================================
# 1.1 灰階與顏色濾鏡
# Hint: B > 100 and B > 1.5 * R and B > 1.2 * G
maskblue1 = (B > 100) & (B > 1.5 * R) & (B > 1.2 * G)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray3x = cv2.merge([gray, gray, gray])

imgres1 = gray3x.copy()
imgres1[maskblue1] = img[maskblue1]

# =======================================
# 1.2 對比與亮度
# Blue: (B > 1.5 * R) and (B > 1.2 * G)
# Yellow: (R > 1.5 * B) and (G > 1.5 * B)
maskblue2 = (B > 1.5 * R) & (B > 1.2 * G)
maskyellow = (R > 1.5 * B) & (G > 1.5 * B)

finmask = maskblue2 | maskyellow

contrast = 50
brightness = 20

imgres2flt = img_float.copy()

# 套用公式
newpix = (imgres2flt[finmask] - 127) * (contrast / 127.0 + 1) + 127 + brightness
imgres2flt[finmask] = newpix

imgres2 = np.clip(imgres2flt, 0, 255).astype(np.uint8)

# =======================================
# 儲存結果至 ./output 資料夾
output_path1 = os.path.join("./output", "q1_1.jpg")
output_path2 = os.path.join("./output", "q1_2.jpg")

cv2.imwrite(output_path1, imgres1)
cv2.imwrite(output_path2, imgres2)

print(f"處理完成！已輸出至：\n  - {output_path1}\n  - {output_path2}")