import cv2
import cv2.aruco as aruco
import numpy as np
import time
import math
import logging
import threading
from djitellopy import Tello

# 抑制 djitellopy 發送 RC 指令時的終端機洗版 log
logging.getLogger("djitellopy").setLevel(logging.WARNING)

# 飛行與狀態機旗標
is_flying = False
current_state = 0  # 0: 等待起飛/鎖定 ID 1, 1: 逼近 ID 1, 2: 繞過板子1, 3: 逼近 ID 2, 4: 繞過板子2, 5: 最終向前, 6: 完成
is_executing_maneuver = False

class SimplePID:
    """輕量且具備最低死區補償的 PID 控制器"""
    def __init__(self, kp, ki, kd, min_output=12, max_output=25):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.min_output = min_output  # 克服 Tello 死區的最低輸出
        self.max_output = max_output
        self.prev_error = 0.0
        self.integral = 0.0
        self.prev_time = time.time()

    def update(self, error):
        current_time = time.time()
        dt = current_time - self.prev_time
        if dt <= 0.001:
            dt = 0.001

        self.integral += error * dt
        derivative = (error - self.prev_error) / dt

        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)

        self.prev_error = error
        self.prev_time = current_time

        # 死區補償：若有誤差但輸出過小，補償到最低推力值
        if abs(error) > 3.0:
            if 0 < output < self.min_output:
                output = self.min_output
            elif -self.min_output < output < 0:
                output = -self.min_output

        return int(max(-self.max_output, min(self.max_output, output)))

def async_takeoff(drone):
    global is_flying
    try:
        drone.takeoff()
        is_flying = True
        print("[INFO] Takeoff complete")
    except Exception as e:
        print(f"[Takeoff Error] {e}")

def async_land(drone):
    global is_flying
    try:
        drone.land()
        is_flying = False
        print("[INFO] Land complete")
    except Exception as e:
        print(f"[Land Error] {e}")

def execute_maneuver(drone, actions):
    """執行循序離散避障動作"""
    global is_executing_maneuver, current_state
    is_executing_maneuver = True
    drone.send_rc_control(0, 0, 0, 0)
    time.sleep(0.5)

    for action, dist in actions:
        if not is_flying:
            break
        try:
            print(f"[MANEUVER] Executing {action} {dist} cm...")
            if action == "right":
                drone.move_right(dist)
            elif action == "left":
                drone.move_left(dist)
            elif action == "forward":
                drone.move_forward(dist)
            elif action == "back":
                drone.move_back(dist)
            time.sleep(0.5)
        except Exception as e:
            print(f"[MANEUVER ERROR] {e}")

    drone.send_rc_control(0, 0, 0, 0)
    is_executing_maneuver = False
    current_state += 1
    print(f"[STATE] Successfully transitioned to STATE {current_state}")

def keyboard(drone, key):
    global is_flying, current_state
    fb_speed = 40
    lf_speed = 40
    ud_speed = 50
    degree = 30

    if key == ord("1"):
        if not is_flying:
            threading.Thread(target=async_takeoff, args=(drone,), daemon=True).start()
    elif key == ord("2"):
        if is_flying:
            threading.Thread(target=async_land, args=(drone,), daemon=True).start()
    elif key == ord("3"):
        drone.send_rc_control(0, 0, 0, 0)
        print("stop!!!!")
    elif key == ord("r"):
        current_state = 0
        print("[INFO] State reset to 0")
    elif key == ord("w"):
        drone.send_rc_control(0, fb_speed, 0, 0)
    elif key == ord("s"):
        drone.send_rc_control(0, -fb_speed, 0, 0)
    elif key == ord("a"):
        drone.send_rc_control(-lf_speed, 0, 0, 0)
    elif key == ord("d"):
        drone.send_rc_control(lf_speed, 0, 0, 0)
    elif key == ord("x"):
        drone.send_rc_control(0, 0, ud_speed, 0)
    elif key == ord("z"):
        drone.send_rc_control(0, 0, -ud_speed, 0)
    elif key == ord("c"):
        drone.send_rc_control(0, 0, 0, degree)
    elif key == ord("v"):
        drone.send_rc_control(0, 0, 0, -degree)
    elif key == ord("5"):
        print("Height:", drone.get_height())
    elif key == ord("6"):
        print("Battery:", drone.get_battery())

def main():
    global current_state, is_executing_maneuver

    fs = cv2.FileStorage("calibration_result.xml", cv2.FILE_STORAGE_READ)
    intrinsic = fs.getNode("intrinsic").mat()
    distortion = fs.getNode("distortion").mat()
    fs.release()

    if intrinsic is None or distortion is None:
        print("[ERROR] 無法讀取 calibration_result.xml，請確認檔案與節點名稱！")
        return

    # PID 控制器 (給予適當推力與最大速度)
    x_pid = SimplePID(kp=0.35, ki=0.0, kd=0.05, min_output=12, max_output=25)
    y_pid = SimplePID(kp=0.45, ki=0.0, kd=0.05, min_output=15, max_output=25)
    z_pid = SimplePID(kp=0.35, ki=0.0, kd=0.05, min_output=15, max_output=30)
    yaw_pid = SimplePID(kp=0.30, ki=0.0, kd=0.04, min_output=12, max_output=25)

    marker_length = 15.0  # cm
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

    # 最佳化 ArUco 檢測參數：強化遠距離 (如 ID 2 距離 115cm 以上) 的小尺寸與邊緣對比偵測
    parameters = aruco.DetectorParameters_create()
    parameters.adaptiveThreshWinSizeMin = 3
    parameters.adaptiveThreshWinSizeMax = 23
    parameters.adaptiveThreshWinSizeStep = 5
    parameters.minMarkerPerimeterRate = 0.02

    drone = Tello()
    drone.connect()
    print("Battery:", drone.get_battery())

    drone.streamon()
    time.sleep(2)
    frame_read = drone.get_frame_read()

    print("=" * 45)
    print("Lab 7 v3 系統就緒")
    print("STATE 1 目標: 50±7cm")
    print("STATE 3 目標: 115±10cm (推進至 <=125cm)")
    print("=" * 45)

    target_reached_counter = 0

    try:
        while True:
            frame = frame_read.frame
            if frame is None:
                cv2.waitKey(1)
                continue

            display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(display_frame, cv2.COLOR_BGR2GRAY)

            corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

            lr_speed = 0
            fwd_speed = 0
            up_speed = 0
            yaw_speed = 0
            target_found = False
            current_dist = 0.0

            # 依據狀態決定目標 ID 與設定距離
            target_id = 1 if current_state in [0, 1] else (2 if current_state == 3 else None)
            target_dist = 50.0 if current_state in [0, 1] else 115.0

            if ids is not None and len(corners) > 0 and target_id is not None:
                aruco.drawDetectedMarkers(display_frame, corners, ids)
                rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                    corners, marker_length, intrinsic, distortion
                )

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
                    x_error = tvec[0]
                    y_error = -tvec[1]
                    current_dist = tvec[2]
                    z_error = current_dist - target_dist

                    # 計算偏航角
                    R, _ = cv2.Rodrigues(rvec)
                    Z_prime = np.dot(R, np.array([0, 0, 1]))
                    yaw_angle = math.degrees(math.atan2(Z_prime[0], Z_prime[2]))

                    # PID 計算輸出速度
                    lr_speed = x_pid.update(x_error)
                    up_speed = y_pid.update(y_error)
                    fwd_speed = z_pid.update(z_error)
                    yaw_speed = yaw_pid.update(yaw_angle)

                    # ---------------------------------------------
                    # 判別條件 (依任務寬容度區分 ID 1 與 ID 2)
                    # ---------------------------------------------
                    if current_state == 3:
                        # ID 2：放寬至 ±10cm (或當距離已推進至 125cm 以內) 且橫向偏移在 15cm 內
                        is_in_range = (current_dist <= 125.0) and (abs(x_error) <= 15.0)
                    else:
                        # ID 1：保持在 50cm 附近 (±7cm)
                        is_in_range = (abs(z_error) <= 7.0) and (abs(x_error) <= 10.0)

                    # 軟性累計計數器 (掉幀衰減 1，不粗暴歸零，連續 4 幀達標即確認到位)
                    if is_in_range:
                        target_reached_counter += 1
                    else:
                        target_reached_counter = max(0, target_reached_counter - 1)

                    target_goal_str = "<=125cm (115+-10)" if current_state == 3 else "50+-7cm"
                    cv2.putText(display_frame, f"Tracking ID: {target_id} | Dist: {current_dist:.1f}cm ({target_goal_str})",
                                (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                    cv2.putText(display_frame, f"RC: [L/R:{lr_speed}, FWD:{fwd_speed}, UP:{up_speed}, YAW:{-yaw_speed}]",
                                (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    cv2.putText(display_frame, f"Reached: {target_reached_counter}/4", (20, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    target_reached_counter = max(0, target_reached_counter - 1)
            else:
                target_reached_counter = max(0, target_reached_counter - 1)

            # ---------------------------
            # 狀態機流程控制
            # ---------------------------
            if is_flying and not is_executing_maneuver:
                if current_state == 0 and target_found:
                    print("[STATE] ID 1 Detected! Entering STATE 1 (Tracking to 50cm)")
                    current_state = 1

                elif current_state == 1 and target_reached_counter >= 4:
                    print("[STATE] Reached 50cm from ID 1! Entering STATE 2 (Bypass Right 90cm)")
                    target_reached_counter = 0
                    current_state = 2
                    threading.Thread(target=execute_maneuver, args=(drone, [("right", 90)]), daemon=True).start()

                elif current_state == 3 and target_reached_counter >= 4:
                    print("[STATE] Reached 115cm (within tolerance) from ID 2! Entering STATE 4 (Bypass Left 150cm)")
                    target_reached_counter = 0
                    current_state = 4
                    threading.Thread(target=execute_maneuver, args=(drone, [("left", 150)]), daemon=True).start()

                elif current_state == 5:
                    print("[STATE] Final Leg! Moving forward 125cm and stopping")
                    threading.Thread(target=execute_maneuver, args=(drone, [("forward", 125)]), daemon=True).start()

            # 畫面資訊
            cv2.putText(display_frame, f"State: {current_state} | Maneuver: {is_executing_maneuver}",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
            cv2.imshow("Lab 7 - Drone Tracking & Obstacle Avoidance", display_frame)

            # ---------------------------
            # 鍵盤與 RC 指令發送
            # ---------------------------
            key = cv2.waitKey(1) & 0xFF
            if key != 255 and key != -1:
                if key == 27:
                    threading.Thread(target=async_land, args=(drone,), daemon=True).start()
                    time.sleep(1)
                    break
                keyboard(drone, key)
            else:
                if is_flying and not is_executing_maneuver:
                    if target_found and (current_state in [1, 3]):
                        drone.send_rc_control(lr_speed, fwd_speed, up_speed, -yaw_speed)
                    else:
                        drone.send_rc_control(0, 0, 0, 0)

    except KeyboardInterrupt:
        drone.land()
    finally:
        drone.send_rc_control(0, 0, 0, 0)
        drone.streamoff
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()