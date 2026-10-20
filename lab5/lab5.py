"""
Lab5 - Part 2: Tello EDU + ArUco marker distance measurement

放在 Tello_Video 資料夾內執行:
    python lab5.py            # 偵測 marker 並顯示 x, y, z, id, 真實距離
    python lab5.py --calib    # 用 Tello 鏡頭拍西洋棋盤格做相機校正 (產生 calibration_result.xml)

按鍵:
    q / ESC : 離開
    s       : 儲存目前畫面截圖
    (校正模式) c : 擷取一張棋盤格影像   k : 完成校正並存檔
"""
import argparse
import math
import os
import time

import cv2
import numpy as np
from djitellopy import Tello

MARKER_SIZE = 15            # marker 邊長 (cm)，所以 tvec 的單位就是 cm
CALIB_FILE = "calibration_result.xml"
DICT_TYPE = cv2.aruco.DICT_4X4_250   # 與第一部分投影片相同

# 校正用棋盤格 (內角點數、每格邊長 cm)，請依你列印的棋盤格修改
CHESS_COLS, CHESS_ROWS = 9, 6
CHESS_SQUARE = 2.5


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------
def load_calibration(path=CALIB_FILE):
    """讀取 calibrateCamera.xml，找不到就用粗略的預設值 (距離會不準)。"""
    if os.path.exists(path):
        fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
        intrinsic = fs.getNode("intrinsic").mat()
        distortion = fs.getNode("distortion").mat()
        fs.release()
        print(f"[INFO] 已讀取校正參數: {path}")
        return intrinsic, distortion

    print(f"[WARN] 找不到 {path}，改用預設內參 (請先執行 --calib 校正，否則距離不準)")
    # Tello 畫面約 960x720，焦距這裡只是粗估值
    intrinsic = np.array([[920.0, 0.0, 480.0],
                          [0.0, 920.0, 360.0],
                          [0.0, 0.0, 1.0]])
    distortion = np.zeros((5, 1))
    return intrinsic, distortion


def run_calibration(frame_read):
    objp = np.zeros((CHESS_ROWS * CHESS_COLS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESS_COLS, 0:CHESS_ROWS].T.reshape(-1, 2) * CHESS_SQUARE

    obj_points, img_points = [], []
    img_size = None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    print("[CALIB] 對準棋盤格按 c 擷取 (建議 15~20 張、不同角度)，按 k 完成校正")
    while True:
        frame = frame_read.frame
        if frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img_size = gray.shape[::-1]

        found, corners = cv2.findChessboardCorners(gray, (CHESS_COLS, CHESS_ROWS), None)
        show = frame.copy()
        if found:
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(show, (CHESS_COLS, CHESS_ROWS), corners, found)
        cv2.putText(show, f"captured: {len(obj_points)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow("calibration", show)

        key = cv2.waitKey(30) & 0xFF
        if key == ord("c") and found:
            obj_points.append(objp)
            img_points.append(corners)
            print(f"[CALIB] 已擷取 {len(obj_points)} 張")
        elif key == ord("k"):
            break
        elif key in (ord("q"), 27):
            cv2.destroyAllWindows()
            return

    if len(obj_points) < 5:
        print("[CALIB] 張數太少，至少需要 5 張")
        cv2.destroyAllWindows()
        return

    rms, intrinsic, distortion, _, _ = cv2.calibrateCamera(
        obj_points, img_points, img_size, None, None)
    print(f"[CALIB] RMS reprojection error = {rms:.4f}")
    print("intrinsic =\n", intrinsic)
    print("distortion =", distortion.ravel())

    fs = cv2.FileStorage(CALIB_FILE, cv2.FILE_STORAGE_WRITE)
    fs.write("intrinsic", intrinsic)
    fs.write("distortion", distortion)
    fs.release()
    print(f"[CALIB] 已儲存 {CALIB_FILE}")
    cv2.destroyAllWindows()


# --------------------------------------------------------------------------
# Marker detection + distance
# --------------------------------------------------------------------------
def run_detection(frame_read):
    intrinsic, distortion = load_calibration()
    dictionary = cv2.aruco.Dictionary_get(DICT_TYPE)
    parameters = cv2.aruco.DetectorParameters_create()

    while True:
        frame = frame_read.frame
        if frame is None:
            continue
        frame = frame.copy()

        # Detect markers
        corners, ids, _rejected = cv2.aruco.detectMarkers(
            frame, dictionary, parameters=parameters)

        if ids is not None and len(ids) > 0:
            frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            # Pose estimation (單位同 MARKER_SIZE -> cm)
            rvecs, tvecs, _obj = cv2.aruco.estimatePoseSingleMarkers(
                corners, MARKER_SIZE, intrinsic, distortion)

            for i in range(len(ids)):
                rvec, tvec = rvecs[i], tvecs[i]
                x, y, z = tvec[0]
                dist = math.sqrt(x * x + y * y + z * z)   # 真實距離 (cm)

                cv2.aruco.drawAxis(frame, intrinsic, distortion, rvec, tvec, MARKER_SIZE * 0.5)




                # 文字放在 marker 左上角附近
                cx, cy = corners[i][0][0]
                cx, cy = int(cx), int(cy)
                lines = [
                    f"id: {int(ids[i][0])}",
                    f"x: {x:.2f}  y: {y:.2f}  z: {z:.2f}",
                    f"dist: {dist:.2f} cm",
                ]
                for j, text in enumerate(lines):
                    cv2.putText(frame, text, (cx, max(20, cy - 10 - 22 * (len(lines) - 1 - j))),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("drone", frame)
        key = cv2.waitKey(33) & 0xFF
        if key == ord("s"):   # 存截圖 (報告/demo 用)
            name = time.strftime("shot_%H%M%S.png")
            cv2.imwrite(name, frame)
            print(f"[INFO] saved {name}")
        elif key in (ord("q"), 27):
            break


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", action="store_true", help="執行相機校正模式")
    args = ap.parse_args()

    # Tello
    drone = Tello()
    drone.connect()
    print(f"[INFO] battery: {drone.get_battery()}%")
    drone.streamon()
    frame_read = drone.get_frame_read()
    time.sleep(2)   # 等第一張影像進來

    try:
        if args.calib:
            run_calibration(frame_read)
        else:
            run_detection(frame_read)
    finally:
        cv2.destroyAllWindows()
        drone.streamoff()
        drone.end()


if __name__ == "__main__":
    main()
