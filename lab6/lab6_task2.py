import cv2
import cv2.aruco as aruco
import numpy as np
import time
import math
import logging
import threading
from djitellopy import Tello
from pyimagesearch.pid import PID

# 抑制 djitellopy 發送 RC 指令時的終端機洗版 log
logging.getLogger("djitellopy").setLevel(logging.WARNING)

# 飛行狀態旗標
is_flying = False

def async_takeoff(drone):
    """非阻塞起飛，避免卡住主迴圈導致 OpenCV 視窗當掉"""
    global is_flying
    try:
        drone.takeoff()
        is_flying = True
    except Exception as e:
        print(f"[Takeoff Error] {e}")

def async_land(drone):
    """非阻塞降落"""
    global is_flying
    try:
        drone.land()
        is_flying = False
    except Exception as e:
        print(f"[Land Error] {e}")

def keyboard(drone, key):
    """手動按鍵控制 (具備最高優先權)"""
    global is_flying
    print("key:", key)

    fb_speed = 40   # 前後速度
    lf_speed = 40   # 左右速度
    ud_speed = 50   # 上下速度
    degree = 30     # 旋轉速度

    # 1: 起飛 (使用 Thread 避免畫面凍結)
    if key == ord('1'):
        if not is_flying:
            threading.Thread(target=async_takeoff, args=(drone,), daemon=True).start()
        else:
            print("[INFO] 無人機已在飛行狀態中")

    # 2: 降落
    elif key == ord('2'):
        if is_flying:
            threading.Thread(target=async_land, args=(drone,), daemon=True).start()
        else:
            print("[INFO] 無人機尚未起飛")

    # 3: 緊急停機
    elif key == ord('3'):
        drone.send_rc_control(0, 0, 0, 0)
        print("stop!!!!")

    # 前後移動
    elif key == ord('w'):
        drone.send_rc_control(0, fb_speed, 0, 0)
        print("forward!!!!")
    elif key == ord('s'):
        drone.send_rc_control(0, (-1) * fb_speed, 0, 0)
        print("backward!!!!")

    # 左右平移
    elif key == ord('a'):
        drone.send_rc_control((-1) * lf_speed, 0, 0, 0)
        print("left!!!!")
    elif key == ord('d'):
        drone.send_rc_control(lf_speed, 0, 0, 0)
        print("right!!!!")

    # 上下升降
    elif key == ord('x'):
        drone.send_rc_control(0, 0, ud_speed, 0)
        print("up!!!!")
    elif key == ord('z'):
        drone.send_rc_control(0, 0, (-1) * ud_speed, 0)
        print("down!!!!")

    # 旋轉偏航
    elif key == ord('c'):
        drone.send_rc_control(0, 0, 0, degree)
        print("rotate!!!!")
    elif key == ord('v'):
        drone.send_rc_control(0, 0, 0, (-1) * degree)
        print("counter rotate!!!!")

    # 狀態查詢
    elif key == ord('5'):
        print("Height:", drone.get_height())
    elif key == ord('6'):
        print("Battery:", drone.get_battery())

def clamp(val, limit):
    """限制速度在 [-limit, limit] 之間，防止暴衝失控"""
    return max(-limit, min(limit, val))

def main():
    # ---------------------------
    # 1. 讀取相機校正參數
    # ---------------------------
    fs = cv2.FileStorage("calibration_result.xml", cv2.FILE_STORAGE_READ)
    intrinsic = fs.getNode("intrinsic").mat()
    distortion = fs.getNode("distortion").mat()
    fs.release()

    if intrinsic is None or distortion is None:
        print("[ERROR] 無法讀取 calibration_result.xml，請確認檔案與節點名稱！")
        return

    # ---------------------------
    # 2. 初始化 PID 控制器
    # ---------------------------
    z_pid = PID(kP=0.4, kI=0.0, kD=0.1)
    y_pid = PID(kP=0.5, kI=0.0, kD=0.1)
    yaw_pid = PID(kP=0.3, kI=0.0, kD=0.08)

    z_pid.initialize()
    y_pid.initialize()
    yaw_pid.initialize()

    max_speed_threshold = 30   # 速度上限限制
    target_distance = 100.0    # 目標保持距離 (cm)
    marker_length = 15.0       # Marker 實際邊長 (cm，若量測不同請自行調整)

    # ---------------------------
    # 3. ArUco 字典設定 (4x4)
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
    time.sleep(2)  # 等待影像串流穩定
    frame_read = drone.get_frame_read()

    print("\n================== 操作說明 ==================")
    print("請點擊彈出的【Drone Tracking】視窗以確保按鍵生效！")
    print("鍵盤 1: 起飛 | 鍵盤 2: 降落 | 鍵盤 3: 緊急煞車")
    print("鍵盤 w/s: 前進/後退 | a/d: 左移/右移 | x/z: 上升/下降 | c/v: 旋轉")
    print("ESC: 安全降落並退出程式")
    print("============================================\n")

    last_rc = (0, 0, 0, 0)

    try:
        while True:
            frame = frame_read.frame
            if frame is None:
                # 即使畫面短暫掉幀，依然要呼叫 waitKey 保持視窗響應
                cv2.waitKey(1)
                continue

            display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(display_frame, cv2.COLOR_BGR2GRAY)

            # 偵測 Marker
            corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

            marker_found = False
            fwd_speed = 0
            up_speed = 0
            yaw_speed = 0
            current_distance = 0.0
            yaw_angle = 0.0

            if ids is not None and len(corners) > 0:
                marker_found = True
                aruco.drawDetectedMarkers(display_frame, corners, ids)

                rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                    corners, marker_length, intrinsic, distortion
                )

                rvec = rvecs[0][0]
                tvec = tvecs[0][0]

                # 繪製姿態軸
                aruco.drawAxis(display_frame, intrinsic, distortion, rvec, tvec, 10)

                # 計算誤差
                current_distance = tvec[2]
                z_error = current_distance - target_distance
                y_error = -tvec[1]

                # 計算 Yaw 偏航夾角
                R, _ = cv2.Rodrigues(rvec)
                Z_prime = np.dot(R, np.array([0, 0, 1]))
                yaw_angle = math.degrees(math.atan2(Z_prime[0], Z_prime[2]))

                # PID 更新
                z_update = z_pid.update(z_error, sleep=0)
                y_update = y_pid.update(y_error, sleep=0)
                yaw_update = yaw_pid.update(yaw_angle, sleep=0)

                fwd_speed = clamp(int(z_update), max_speed_threshold)
                up_speed = clamp(int(y_update), max_speed_threshold)
                yaw_speed = clamp(int(yaw_update), max_speed_threshold)

                # 畫面上顯示數值
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
            # 優先權判斷與 RC 控制
            # ---------------------------
            key = cv2.waitKey(1) & 0xFF

            # 【最高優先權】：只要有手動按鍵，直接執行手動控制
            if key != 255 and key != -1:
                if key == 27:  # ESC 鍵降落退出
                    threading.Thread(target=async_land, args=(drone,), daemon=True).start()
                    time.sleep(1)
                    break
                keyboard(drone, key)
                last_rc = None

            # 【第二優先權】：無按鍵輸入時，才執行自動控制
            else:
                if is_flying and marker_found:
                    # 使用 -yaw_speed 修正左右轉向顛倒
                    cmd = (0, fwd_speed, up_speed, -yaw_speed)
                    drone.send_rc_control(*cmd)
                    last_rc = cmd
                else:
                    # 避免重複發送懸停指令
                    if last_rc != (0, 0, 0, 0):
                        drone.send_rc_control(0, 0, 0, 0)
                        last_rc = (0, 0, 0, 0)

    except KeyboardInterrupt:
        print("[INFO] 偵測到強制中斷，執行降落...")
        drone.land()
    finally:
        drone.send_rc_control(0, 0, 0, 0)
        drone.streamoff()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()