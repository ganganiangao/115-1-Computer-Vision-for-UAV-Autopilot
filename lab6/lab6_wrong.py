import cv2
import cv2.aruco as aruco
import numpy as np
import time
import math
from djitellopy import Tello
import logging
from pyimagesearch.pid import PID
from keyboard_djitellopy import keyboard

logging.getLogger("djitellopy").setLevel(logging.WARNING)

def clamp(val, limit):
    """限制速度在 [-limit, limit] 之間，防止暴衝失控"""
    return max(-limit, min(limit, val))

def main():
    # ---------------------------
    # 1. 讀取相機校正參數 (XML 檔)
    # ---------------------------
    fs = cv2.FileStorage("calibration_result.xml", cv2.FILE_STORAGE_READ)
    intrinsic = fs.getNode("intrinsic").mat()
    distortion = fs.getNode("distortion").mat()
    fs.release()

    if intrinsic is None or distortion is None:
        print("[ERROR] 無法讀取 calibrateCamera.xml，請確認檔案路徑！")
        return

    # ---------------------------
    # 2. 初始化 PID 控制器
    # ---------------------------
    # z_pid: 控制前後距離 (Z 軸)
    # y_pid: 控制上下高度 (Y 軸)
    # yaw_pid: 控制轉向角度 (Yaw)
    z_pid = PID(kP=0.7, kI=0.0001, kD=0.1)
    y_pid = PID(kP=0.7, kI=0.0001, kD=0.1)
    yaw_pid = PID(kP=0.7, kI=0.0001, kD=0.1)

    z_pid.initialize()
    y_pid.initialize()
    yaw_pid.initialize()

    # 最大速度上限（投影片建議限制速度防止暴衝）
    max_speed_threshold = 30
    # 目標保持距離 (公分，投影片設定目標約 100 cm)
    target_distance = 100.0
    # Marker 實際邊長 (公分，若實際大小不同請自行修改，例如 15.0 cm)
    marker_length = 15.0

    # ---------------------------
    # 3. ArUco 字典與偵測器設定 (4x4)
    # ---------------------------
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters_create()

    # ---------------------------
    # 4. Tello 初始化與影像串流
    # ---------------------------
    drone = Tello()
    drone.connect()
    print("Battery:", drone.get_battery())

    drone.streamon()
    time.sleep(2)  # 等待影像串流穩定建立
    frame_read = drone.get_frame_read()

    print("\n--- 操作說明 ---")
    print("鍵盤 1: 起飛 | 鍵盤 2: 降落 | 鍵盤 3: 緊急停機")
    print("鍵盤 w/s: 前進/後退 | a/d: 左移/右移 | x/z: 上升/下降 | c/v: 旋轉")
    print("ESC: 降落並退出程式")
    print("手動按鍵具備最高優先權，鬆開按鍵且偵測到 Marker 時切換為自動跟隨\n")

    try:
        while True:
            frame = frame_read.frame
            if frame is None:
                continue

            # OpenCV 預設為 BGR，若 djitellopy 讀取為 RGB，可轉為 BGR 供顯示
            display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(display_frame, cv2.COLOR_BGR2GRAY)

            # 偵測 ArUco Marker
            corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

            marker_found = False
            fwd_speed = 0
            up_speed = 0
            yaw_speed = 0

            if ids is not None and len(corners) > 0:
                marker_found = True
                aruco.drawDetectedMarkers(display_frame, corners, ids)

                # 估計位姿 (rvec, tvec)
                rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                    corners, marker_length, intrinsic, distortion
                )

                rvec = rvecs[0][0]
                tvec = tvecs[0][0]  # [X, Y, Z]

                # 繪製姿態座標軸 (長度 10 cm)
                aruco.drawAxis(display_frame, intrinsic, distortion, rvec, tvec, 10)

                # ---------------------------
                # 計算各軸誤差 (投影片第 8, 9, 15, 18 頁)
                # ---------------------------
                current_distance = tvec[2]

                # 1. 前後誤差 (Z 軸)：目前距離 - 目標距離
                # 若距離大於 100 cm，z_error > 0，無人機需前進拉近
                z_error = current_distance - target_distance

                # 2. 上下誤差 (Y 軸)：相機 Y 軸朝下，Marker 偏上時 tvec[1] < 0，因此誤差取負
                y_error = -tvec[1]

                # 3. 旋轉偏航夾角 (Yaw，投影片第 18 頁公式)
                R, _ = cv2.Rodrigues(rvec)  # 轉為旋轉矩陣
                Z_prime = np.dot(R, np.array([0, 0, 1]))  # R 乘以 Z 軸向量 (0, 0, 1)
                # 投影到 XZ 平面並求夾角 (度數)
                yaw_angle = math.degrees(math.atan2(Z_prime[0], Z_prime[2]))

                # ---------------------------
                # PID 平滑更新
                # ---------------------------
                z_update = z_pid.update(z_error, sleep=0)
                y_update = y_pid.update(y_error, sleep=0)
                yaw_update = yaw_pid.update(yaw_angle, sleep=0)

                # 限制速度大小並轉為整數
                fwd_speed = clamp(int(z_update), max_speed_threshold)
                up_speed = clamp(int(y_update), max_speed_threshold)
                yaw_speed = clamp(int(yaw_update), max_speed_threshold)

                # 於畫面上顯示目前距離與偏航角 (驗收要求)
                cv2.putText(display_frame, f"Dist: {current_distance:.1f} cm", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Yaw: {yaw_angle:.1f} deg", (20, 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(display_frame, "Marker NOT Found", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # 顯示畫面
            cv2.imshow("Drone Tracking", display_frame)

            # ---------------------------
            # 優先權判斷與控制指令發送
            # ---------------------------
            key = cv2.waitKey(1) & 0xFF

            # 【最高優先權】：只要檢測到鍵盤按下，立刻執行手動控制
            if key != 255 and key != -1:
                if key == 27:  # ESC 鍵安全降落退出
                    drone.land()
                    break
                keyboard(drone, key)

            # 【第二優先權】：無按鍵輸入時，才執行自動控制
            else:
                if marker_found:
                    # send_rc_control(左右, 前後, 上下, 旋轉)
                    drone.send_rc_control(0, fwd_speed, up_speed, yaw_speed)
                else:
                    # 沒按鍵也沒偵測到 Marker，原地懸停防止漂移
                    drone.send_rc_control(0, 0, 0, 0)

    except KeyboardInterrupt:
        print("[INFO] 中斷程式，準備降落...")
        drone.emergency()
    finally:
        drone.send_rc_control(0, 0, 0, 0)
        drone.streamoff()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()