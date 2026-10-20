import cv2
import numpy as np
import time
from djitellopy import Tello

"""
cv2.findChessboardCorners 找粗略的
cv2.drawChessboardCorners 畫粗略的
cv2.cornerSubPix 細節微調
cv2.calibrateCamera 算相機參數
cv2.FileStorage 儲存成 XML
"""

# 簡報指定規格：棋盤格內部角點維度 (columns=9, rows=6)
PATTERN_SIZE = (9, 6)

# 準備物體座標 (3D object points, Z=0)
objp = np.zeros((PATTERN_SIZE[0] * PATTERN_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:PATTERN_SIZE[0], 0:PATTERN_SIZE[1]].T.reshape(-1, 2)

objpoints = []  # 儲存 3D 世界座標
imgpoints = []  # 儲存 2D 像素座標

# cornerSubPix 終止條件
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)

drone = Tello()
drone.connect()
print(f"battery: {drone.get_battery()}%")
drone.streamon()
frame_read = drone.get_frame_read()
time.sleep(2)   # 等第一張影像進來


print("操作說明：")
print("  - 按 's' 鍵：儲存當前成功偵測到角點的畫面")
print("  - 按 'c' 鍵：當儲存影像 >= 4 張時，計算參數並存成 XML")
print("  - 按 'ESC' 鍵：退出程式")

gray_shape = None

while True:
    frame = frame_read.frame
    if frame is None:
        continue
    frame = frame.copy()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_shape = gray.shape[::-1]
    display_frame = frame.copy()

    # 偵測棋盤格角點
    # 若 54 個內部角點全部找到，found 為 True，並回傳粗略角點座標 corners
    found, corners = cv2.findChessboardCorners(gray, PATTERN_SIZE, None)

    if found:
        # 繪製角點供即時預覽(彩色的那個)
        cv2.drawChessboardCorners(display_frame, PATTERN_SIZE, corners, found)

    # 顯示目前已採樣張數
    cv2.putText(
        display_frame,
        f"Saved Frames: {len(imgpoints)} (Need >= 4)",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Task 1 - Camera Calibration", display_frame)
    key = cv2.waitKey(33) & 0xFF

    if key == ord('s'):
        if found:
            # 次像素精度微調
            # 利用局部影像梯度進行插值微調，將剛剛找到的粗略整數角點優化至亞像素（小數點精度）等級
            refined_corners = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria
            )
            objpoints.append(objp)  # 存入一份理想的 3D 世界座標 objp
            imgpoints.append(refined_corners)  # 存入剛剛優化完的 2D 像素座標 refined_corners
            print(f"已記錄第 {len(imgpoints)} 張有效視角！")
        else:
            print("目前畫面未偵測到完整的棋盤格，無法記錄！")

    elif key == ord('c'):
        if len(imgpoints) >= 4:
            print("開始計算相機參數...")
            # cv2.calibrateCamera(...) 校正核心演算法，將所有採樣的 3D 座標與 2D 像素座標輸入求解
            # mtx (cameraMatrix) : 相機內部參數矩陣（包含焦距 fx, fy 與主點光心 (cx, cy)
            # dist (distCoeffs): 鏡頭的畸變係數（包含徑向與切向畸變 k1, k2, k3, p1, p2)
            # rvecs, tvecs: 相機對於每一張棋盤格的旋轉與平移矩陣（外部參數）
            ret_calib, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, imgpoints, gray_shape, None, None
            )

            print("\n----- 相機內部參數 (Camera Matrix) -----")
            print(mtx)
            print("\n----- 畸變係數 (Distortion Coefficients) -----")
            print(dist)

            # 儲存為 XML 檔
            fs = cv2.FileStorage("calibration_result.xml", cv2.FILE_STORAGE_WRITE)
            fs.write("intrinsic", mtx)
            fs.write("distortion", dist)
            fs.release()
            print("\n成功將參數寫入 calibration_result.xml！")
            break
        else:
            print(f"目前只有 {len(imgpoints)} 張，至少需要 4 張影像才能開始計算！")

    elif key == 27:  # ESC
        break

cv2.destroyAllWindows()
drone.streamoff()
drone.end()

