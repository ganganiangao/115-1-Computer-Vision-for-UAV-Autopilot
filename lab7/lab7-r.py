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

# 飛行與狀態機旗標
is_flying = False
current_state = 0  # 0: 等待起飛/鎖定 ID 1, 1: 逼近 ID 1, 2: 繞過板子1, 3: 逼近 ID 2, 4: 繞過板子2, 5: 最終向前, 6: 完成懸停/降落
is_executing_maneuver = False  # 避免重複觸發離散動作執行續

def async_takeoff(drone):
    """非阻塞起飛"""
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

def execute_maneuver(drone, actions):
    """
    在獨立執行緒中執行循序的離散移動指令 (例如繞行、定向位移)
    actions: list of tuples -> (func_name, arg)
    """
    global is_executing_maneuver, current_state
    is_executing_maneuver = True
    drone.send_rc_control(0, 0, 0, 0)
    time.sleep(0.5)

    for action, dist in actions:
        if not is_flying:
            break
        try:
            print(f"[MANEUVER] Executing {action} {dist} cm...")
            if action == 'right':
                drone.move_right(dist)
            elif action == 'left':
                drone.move_left(dist)
            elif action == 'forward':
                drone.move_forward(dist)
            elif action == 'back':
                drone.move_back(dist)
            time.sleep(0.5)
        except Exception as e:
            print(f"[MANEUVER ERROR] {e}")

    drone.send_rc_control(0, 0, 0, 0)
    is_executing_maneuver = False
    current_state += 1
    print(f"[STATE] Transitioned to STATE {current_state}")

def keyboard(drone, key):
    """手動按鍵控制 (具備最高優先權)"""
    global is_flying, current_state
    print("key:", key)

    fb_speed = 40
    lf_speed = 40
    ud_speed = 50
    degree = 30

    if key == ord('1'):
        if not is_flying:
            threading.Thread(target=async_takeoff, args=(drone,), daemon=True).start()
        else:
            print("[INFO] 無人機已在飛行狀態中")

    elif key == ord('2'):
        if is_flying:
            threading.Thread(target=async_land, args=(drone,), daemon=True).start()
        else:
            print("[INFO] 無人機尚未起飛")

    elif key == ord('3'):
        drone.send_rc_control(0, 0, 0, 0)
        print("stop!!!!")

    # 手動重設/推進狀態 (除錯輔助)
    elif key == ord('r'):
        current_state = 0
        print("[INFO] State reset to 0")

    elif key == ord('w'):
        drone.send_rc_control(0, fb_speed, 0, 0)
    elif key == ord('s'):
        drone.send_rc_control(0, -fb_speed, 0, 0)
    elif key == ord('a'):
        drone.send_rc_control(-lf_speed, 0, 0, 0)
    elif key == ord('d'):
        drone.send_rc_control(lf_speed, 0, 0, 0)
    elif key == ord('x'):
        drone.send_rc_control(0, 0, ud_speed, 0)
    elif key == ord('z'):
        drone.send_rc_control(0, 0, -ud_speed, 0)
    elif key == ord('c'):
        drone.send_rc_control(0, 0, 0, degree)
    elif key == ord('v'):
        drone.send_rc_control(0, 0, 0, -degree)
    elif key == ord('5'):
        print("Height:", drone.get_height())
    elif key == ord('6'):
        print("Battery:", drone.get_battery())

def clamp(val, limit):
    """限制速度在 [-limit, limit] 之間"""
    return max(-limit, min(limit, val))

def main():
    global current_state, is_executing_maneuver

    # 1. 讀取相機校正參數
    fs = cv2.FileStorage("calibration_result.xml", cv2.FILE_STORAGE_READ)
    intrinsic = fs.getNode("intrinsic").mat()
    distortion = fs.getNode("distortion").mat()
    fs.release()

    if intrinsic is None or distortion is None:
        print("[ERROR] 無法讀取 calibration_result.xml，請確認檔案與節點名稱！")
        return

    # 2. 初始化 PID 控制器 (新增 x_pid 控制左右平移使 Marker 置中)
    x_pid = PID(kP=0.4, kI=0.0, kD=0.1)
    y_pid = PID(kP=0.5, kI=0.0, kD=0.1)
    z_pid = PID(kP=0.4, kI=0.0, kD=0.1)
    yaw_pid = PID(kP=0.3, kI=0.0, kD=0.08)

    x_pid.initialize()
    y_pid.initialize()
    z_pid.initialize()
    yaw_pid.initialize()

    max_speed_threshold = 25   # 避障場地建議放緩速度
    marker_length = 15.0       # Marker 實際邊長 (cm)

    # 3. ArUco 字典設定
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters_create()

    # 4. Tello 初始化
    drone = Tello()
    drone.connect()
    print("Battery:", drone.get_battery())

    drone.streamon()
    time.sleep(2)
    frame_read = drone.get_frame_read()

    print("""
================== 操作與狀態說明 ==================
鍵盤 1: 起飛 | 鍵盤 2: 降落 | 鍵盤 3: 緊急煞車
鍵盤 r: 重設狀態機至 State 0
ESC: 安全降落並退出程式

STATE 0: 起飛等待/尋找 ID 1
STATE 1: 追蹤並逼近 ID 1 (目標距離 50 cm)
STATE 2: 看到 1 向右繞過板子 1 (右移 90 cm)
STATE 3: 追蹤並逼近 ID 2 (目標距離 115 cm)
STATE 4: 看到 2 向左繞過板子 2 (左移 150 cm)
STATE 5: 穿越並向前 125 cm 後停下
STATE 6: 任務完成 (懸停或按 2 降落)
==================================================
""")

    last_rc = (0, 0, 0, 0)
    target_reached_counter = 0  # 確保穩定停留於目標距離的計數器

    try:
        while True:
            frame = frame_read.frame
            if frame is None:
                cv2.waitKey(1)
                continue

            display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(display_frame, cv2.COLOR_BGR2GRAY)

            # 偵測 Marker
            corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

            lr_speed = 0
            fwd_speed = 0
            up_speed = 0
            yaw_speed = 0
            target_found = False
            current_dist = 0.0

            # 根據當前狀態篩選目標 ID
            target_id = None
            target_dist = None

            if current_state in [0, 1]:
                target_id = 1
                target_dist = 50.0   # ID 1 終止距離 50 cm
            elif current_state == 3:
                target_id = 2
                target_dist = 115.0  # ID 2 終止距離 115 cm

            # Marker 位姿估計與 PID 計算
            if ids is not None and len(corners) > 0:
                aruco.drawDetectedMarkers(display_frame, corners, ids)
                rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                    corners, marker_length, intrinsic, distortion
                )

                # 尋找目前狀態指定的 target_id
                target_idx = -1
                for idx, marker_id in enumerate(ids.flatten()):
                    if marker_id == target_id:
                        target_idx = idx
                        break

                if target_idx != -1:
                    target_found = True
                    rvec = rvecs[target_idx][0]
                    tvec = tvecs[target_idx][0]
                    aruco.drawAxis(display_frame, intrinsic, distortion, rvec, tvec, 10)

                    # 誤差計算
                    x_error = tvec[0]              # 左右橫向偏移
                    y_error = -tvec[1]             # 上下偏移
                    current_dist = tvec[2]         # 前後距離
                    z_error = current_dist - target_dist

                    # 計算 Yaw 角度
                    R, _ = cv2.Rodrigues(rvec)
                    Z_prime = np.dot(R, np.array([0, 0, 1]))
                    yaw_angle = math.degrees(math.atan2(Z_prime[0], Z_prime[2]))

                    # PID 更新
                    x_update = x_pid.update(x_error, sleep=0)
                    y_update = y_pid.update(y_error, sleep=0)
                    z_update = z_pid.update(z_error, sleep=0)
                    yaw_update = yaw_pid.update(yaw_angle, sleep=0)

                    lr_speed = clamp(int(x_update), max_speed_threshold)
                    up_speed = clamp(int(y_update), max_speed_threshold)
                    fwd_speed = clamp(int(z_update), max_speed_threshold)
                    yaw_speed = clamp(int(yaw_update), max_speed_threshold)

                    # 狀態切換判定 (連續 5 幀穩定在誤差範圍內即視為到位)
                    if abs(z_error) < 6.0 and abs(x_error) < 8.0:
                        target_reached_counter += 1
                    else:
                        target_reached_counter = max(0, target_reached_counter - 1)

                    cv2.putText(display_frame, f"Tracking ID: {target_id} | Dist: {current_dist:.1f} cm", 
                                (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(display_frame, f"Reached count: {target_reached_counter}/5", 
                                (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    target_reached_counter = 0
            else:
                target_reached_counter = 0

            # ---------------------------
            # 狀態機轉移與動作排程
            # ---------------------------
            if is_flying and not is_executing_maneuver:
                if current_state == 0 and target_found:
                    print("[STATE] ID 1 Detected! Entering STATE 1 (Tracking to 50cm)")
                    current_state = 1

                elif current_state == 1 and target_reached_counter >= 5:
                    print("[STATE] Reached 50cm from ID 1! Entering STATE 2 (Bypass Obstacle 1)")
                    target_reached_counter = 0
                    current_state = 2
                    # 看到 1 向右避障 (右移 90 cm)
                    threading.Thread(
                        target=execute_maneuver, 
                        args=(drone, [('right', 90)]), 
                        daemon=True
                    ).start()

                elif current_state == 3 and target_reached_counter >= 5:
                    print("[STATE] Reached 115cm from ID 2! Entering STATE 4 (Bypass Obstacle 2)")
                    target_reached_counter = 0
                    current_state = 4
                    # 看到 2 向左避障 (左移 150 cm)
                    threading.Thread(
                        target=execute_maneuver, 
                        args=(drone, [('left', 150)]), 
                        daemon=True
                    ).start()

                elif current_state == 5:
                    print("[STATE] Final Leg! Moving forward 125cm and stopping")
                    threading.Thread(
                        target=execute_maneuver, 
                        args=(drone, [('forward', 125)]), 
                        daemon=True
                    ).start()

            # 繪製狀態資訊
            cv2.putText(display_frame, f"State: {current_state} | Maneuver: {is_executing_maneuver}", 
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)

            cv2.imshow("Lab 7 - Drone Tracking & Obstacle Avoidance", display_frame)

            # ---------------------------
            # 優先權判斷與 RC 控制
            # ---------------------------
            key = cv2.waitKey(1) & 0xFF

            # 【最高優先權】：手動按鍵覆蓋
            if key != 255 and key != -1:
                if key == 27:  # ESC 鍵安全降落
                    threading.Thread(target=async_land, args=(drone,), daemon=True).start()
                    time.sleep(1)
                    break
                keyboard(drone, key)
                last_rc = None

            # 【第二優先權】：自動控制 (若在執行離散位移 maneuver 則不發 RC)
            else:
                if is_flying and not is_executing_maneuver:
                    if target_found and (current_state in [1, 3]):
                        # 控制：x 左右平移置中, y 上下, z 前後, yaw 偏航旋轉
                        cmd = (lr_speed, fwd_speed, up_speed, -yaw_speed)
                        drone.send_rc_control(*cmd)
                        last_rc = cmd
                    else:
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