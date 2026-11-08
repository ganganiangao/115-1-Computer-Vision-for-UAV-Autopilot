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

# ==========================================
# 賽道參數微調區 (依 EAB06 現場實際高度與距離微調)
# ==========================================
# 第一段參數
ID1_TARGET_DIST = 50.0       # ID 1 目標距離 (cm)
ID1_BYPASS_RIGHT = 90        # 繞過 ID 1 向右距離 (cm)
ID2_TARGET_DIST = 115.0      # ID 2 目標距離 (cm)
ID2_INIT_LEFT = 60           # 看到 ID 2 到位後，先盲左移脫離板子的距離 (cm)
SCAN_ID0_LEFT_SPEED = -28    # 持續往左找 ID 0 的水平速度 (-100~100)

# 第二段參數 (手持 ID 0)
ID0_FOLLOW_DIST = 75.0       # 跟隨手持 ID 0 的保持距離 (cm)
ID3_HANDOFF_DIST = 170.0     # 當看到 ID 3 且距離小於此值時，自動脫離 ID 0 切入第三段

# 第三段參數 (ID 3 鑽桌子 -> 轉向找 ID 4)
ID3_TARGET_DIST = 80.0       # 在桌前對準 ID 3 的準備鑽桌距離 (cm)
TABLE_DOWN_DIST = 55         # 鑽桌子下降高度 (cm) - 注意 Tello 最低離地限制
TABLE_FWD_DIST = ID3_TARGET_DIST + 180 # 穿過桌底向前距離 (cm) (規格為 150cm，實際上是 200cm)
TABLE_CW_DEG = 90            # 穿出桌底後向右旋轉角度 (度)
TABLE_BACK_DIST = 50           # 旋轉後先往後方便找 ID 4 (cm)
TABLE_UP_DIST = 135           # 旋轉後上升找 ID 4 的高度 (cm)

# 第四段參數 (ID 4 -> 越過板子上方 -> 找 ID 5)
SCAN_ID4_UP_SPEED = 20      # 向上尋找 ID 4 的速度
SCAN_ID4_BACK_SPEED = -12    # 向上尋找 ID 4 時向後擴展視野的速度
ID4_TARGET_DIST = 80.0      # 與牆上 ID 4 保持的距離 (cm)
OVER_BOARD_UP_DIST = 0      # 對準 ID 4 後額外微幅上升高度，確保能從上方越過板子 (cm)
BACK_FROM_ID4_DIST = 30      # 對準 ID 4 後向後距離，拓展視野方便尋找 ID 5
SCAN_ID5_LEFT_SPEED = -20    # 越過板子上方往左找 ID 5 的水平速度

# 第五段參數 (ID 5 後退至 150cm 降落桌上)
ID5_TARGET_DIST = 80.0        # 與牆上 ID 5 保持的距離 (cm)
DESIRED_LAND_DIST = 155.0     # 桌面中心距離牆壁 ID 5 的實際距離 (cm)

# ==========================================
# 狀態機定義
# ==========================================
# 0: 等待起飛 / 尋找 ID 1
# 1: 逼近 ID 1 至 50cm
# 2: 執行向右避障 90cm (Maneuver)
# 3: 逼近 ID 2 至 115cm & 左移 60cm 脫離板子(Maneuver)
# 4: 持續掃描 ID 0
# 5: 跟隨手持 ID 0 繞椅子 (若同時看到 ID 3 進入範圍則自動切換至 State 6)
# 6: 對準桌子上的 ID 3 (距離 80cm)
# 7: 執行鑽桌子與旋轉上升複合動作 (Maneuver: 下 -> 前 dist_3 + 200 -> 右轉 90 -> 上)
# 8: 對準並逼近牆面 ID 4 至 80cm，到位後後退 30cm 拓展視野 (Maneuver) + 可能要先向上跨越板子 (Maneuver)
# 9: 持續向左水平移動直到看見 ID 5  註記：先測試看看單純向左可不可以找到 ID 5 如果不行就要向後擴大視野
# 10: 對準 ID 5 (距離 80 cm) 並後退至距離 150cm (桌子正上方)
# 11: 降落於桌上，賽道完成

is_flying = False
current_state = 0
is_executing_maneuver = False

""" 輕量且具備最低死區補償的 PID 控制器"""
class SimplePID:  
    def __init__(self, kp, ki, kd, min_output=12, max_output=28, deadband=5.0, d_alpha=0.3):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.min_output = min_output
        self.max_output = max_output
        self.deadband = deadband      # 誤差小於此值視為到位 → 輸出 0（必須小於各 state 的 tolerance，否則會在容許範圍內持續被推而過衝）
        self.d_alpha = d_alpha        # 微分項低通濾波係數 (0~1，越小越平滑)，抑制 ArUco 距離/角度雜訊被放大
        self.reset()

    # 在成功轉入下一個狀態時（例如看到 ID 0、ID 3 或 ID 5），執行 .reset() 清空內部累積狀態，確保前一個動作的慣性不干擾新動作
    def reset(self):
        self.prev_error = 0.0
        self.integral = 0.0
        self.d_filt = 0.0
        self.first = True
        self.prev_time = time.time()

    def update(self, error, custom_max=None):
        current_time = time.time()
        dt = max(current_time - self.prev_time, 0.001)
        self.prev_time = current_time

        # 第一次更新 或 丟失目標超過 0.5 秒後重新看到：不計算微分，避免 derivative kick (第一幀 error/dt 暴衝)
        if self.first or dt > 0.5:
            self.first = False
            self.d_filt = 0.0
        else:
            raw_d = (error - self.prev_error) / dt
            self.d_filt = self.d_alpha * raw_d + (1 - self.d_alpha) * self.d_filt
        self.prev_error = error

        # 死區：已到位就不再推，讓無人機穩定停在容許範圍內
        if abs(error) <= self.deadband:
            return 0

        self.integral += error * dt
        # 限制積分飽和
        self.integral = max(-50.0, min(50.0, self.integral))

        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * self.d_filt)

        # 最低輸出補償 (克服 Tello 小速度指令無反應的死區)
        if 0 < abs(output) < self.min_output:
            output = math.copysign(self.min_output, output)

        max_lim = custom_max if custom_max is not None else self.max_output
        return int(max(-max_lim, min(max_lim, output)))


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
        drone.send_rc_control(0, 0, 0, 0)
        time.sleep(0.2)
        drone.land()
        is_flying = False
        print("[INFO] Land complete")
    except Exception as e:
        print(f"[Land Error] {e}")


"""
    執行循序離散動作 (為求加快速度，將動作間的硬性停頓壓縮至 0.25 秒)
    支援指令: right, left, forward, back, up, down, cw, ccw
    專門處理那些「無法靠看 Marker 即時對準，必須照固定距離或角度硬走」的過渡動作（例如：繞過板子、鑽桌底向前衝、旋轉 90 度等）
"""
def execute_maneuver(drone, actions, next_state=None):
    
    global is_executing_maneuver, current_state
    is_executing_maneuver = True
    drone.send_rc_control(0, 0, 0, 0)
    time.sleep(0.25)

    for action, val in actions:
        if not is_flying:
            break
        # -------------------------------------------------------------
        # Tello SDK 參數合法性防呆過濾
        # 1. 旋轉 (cw, ccw): 範圍 1 ~ 360 度
        # 2. 位移 (up, down, left, right, forward, back): 範圍 20 ~ 500 cm
        # -------------------------------------------------------------
        if action in ["cw", "ccw"]:
            if val < 1:
                print(f"[MANEUVER WARN] {action} 旋轉角度 {val} < 1 度，略過此動作")
                continue
        else:
            if val < 20:
                print(f"[MANEUVER WARN] {action} 位移距離 {val} < 20 cm (SDK 底線限制)，略過此動作")
                continue

        try:
            print(f"[MANEUVER] Executing {action} {val}...")
            if action == "right":
                drone.move_right(val)
            elif action == "left":
                drone.move_left(val)
            elif action == "forward":
                drone.move_forward(val)
            elif action == "back":
                drone.move_back(val)
            elif action == "up":
                drone.move_up(val)
            elif action == "down":
                drone.move_down(val)
            elif action == "cw":
                drone.rotate_clockwise(val)
            elif action == "ccw":
                drone.rotate_counter_clockwise(val)
            time.sleep(0.25)
        except Exception as e:
            print(f"[MANEUVER ERROR] {e}")

    drone.send_rc_control(0, 0, 0, 0)
    if next_state is not None:
        current_state = next_state
    else:
        current_state += 1
    is_executing_maneuver = False
    print(f"[STATE] Successfully transitioned to STATE {current_state}")

"""
#  手動控制
=========================== [ 快捷鍵速查表 ] ===========================
  【飛行狀態】          【鍵盤佈局對照】              【賽道除錯救援】
   [1] 起飛                [w] 前進                    [r] 狀態重置為 0
   [2] 降落        [a] 左移         [d] 右移           [b] 狀態退回上一關
   [3] 急停懸停            [s] 後退                    [n] 強制跳下一關
   [ESC] 緊降退出
  ---------------------------------------------------------------------
  【高度與轉向】        【狀態查詢】
   [x] 上升              [5] 查詢高度
   [z] 下降              [6] 查詢電量
   [c] 順時針轉 45°(右轉)
   [v] 逆時針轉 45°(左轉)
=======================================================================
"""
def keyboard(drone, key):
    global is_flying, current_state
    fb_speed = 45
    lf_speed = 45
    ud_speed = 50
    degree = 45

    if key == ord("1"):
        if not is_flying:
            threading.Thread(target=async_takeoff, args=(drone,), daemon=True).start()
    elif key == ord("2"):
        if is_flying:
            threading.Thread(target=async_land, args=(drone,), daemon=True).start()
    elif key == ord("3"):
        drone.send_rc_control(0, 0, 0, 0)
        print("[MANUAL] Emergency Hover Stop!")
    elif key == ord("r"):
        current_state = 0
        print("[INFO] State reset to 0")
    elif key == ord("n"):
        # 測試與救援神鍵：強制跳過當前狀態進入下一階段
        current_state += 1
        print(f"[MANUAL] Forced jump to STATE {current_state}")
    elif key == ord("b"):
        # 退回上一狀態
        current_state = max(0, current_state - 1)
        print(f"[MANUAL] Forced back to STATE {current_state}")
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

"""從偵測到的所有 Markers 中精準抽出指定 target_id 的姿態與資訊"""
def get_marker_pose(corners, ids, target_id, marker_length, intrinsic, distortion):
    
    if ids is None or len(corners) == 0:
        return False, None, None, 0.0, 0.0, 0.0, 0.0

    flat_ids = ids.flatten()
    if target_id not in flat_ids:
        return False, None, None, 0.0, 0.0, 0.0, 0.0

    idx = list(flat_ids).index(target_id)
    rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
        corners, marker_length, intrinsic, distortion
    )
    rvec = rvecs[idx][0]
    tvec = tvecs[idx][0]

    x_err = tvec[0]
    y_err = -tvec[1]
    dist = tvec[2]

    R, _ = cv2.Rodrigues(rvec)
    Z_prime = np.dot(R, np.array([0, 0, 1]))
    # ArUco 的 Z 軸由 marker 指向鏡頭，正對時 Z_prime ≈ (0, 0, -1)
    # 原本 atan2(Z[0], Z[2]) 正對時 ≈ ±180°，導致 yaw PID 永遠飽和 (bang-bang 抖動)
    # 改為 atan2(Z[0], -Z[2])：正對時 ≈ 0°，機頭偏右 (cw) 為正
    yaw_deg = math.degrees(math.atan2(Z_prime[0], -Z_prime[2]))

    # 回傳是否找到目標（True/False），
    # 以及繪圖與 PID 所需的 4 個直接控制變數：
    # 水平偏量 (x_err)、高度偏量 (y_err)、前方距離 (dist)、偏航角度 (yaw_deg)
    return True, rvec, tvec, x_err, y_err, dist, yaw_deg


def main():
    global current_state, is_executing_maneuver

    fs = cv2.FileStorage("calibration_result.xml", cv2.FILE_STORAGE_READ)
    intrinsic = fs.getNode("intrinsic").mat()
    distortion = fs.getNode("distortion").mat()
    fs.release()

    if intrinsic is None or distortion is None:
        print("[ERROR] 無法讀取 calibration_result.xml，請確認檔案路徑！")
        return

    # PID 控制器 (微調提高上限以加快賽道收斂速度)
    # deadband 需小於各 state 的 tolerance (目前 x/z ±15cm、yaw ±15°)
    x_pid = SimplePID(kp=0.38, ki=0.0, kd=0.06, min_output=12, max_output=28, deadband=5.0)
    y_pid = SimplePID(kp=0.48, ki=0.0, kd=0.05, min_output=15, max_output=28, deadband=6.0)
    z_pid = SimplePID(kp=0.40, ki=0.0, kd=0.06, min_output=15, max_output=35, deadband=5.0)
    # yaw 誤差修正後為小角度 (0 附近)，kp 需調大才有足夠推力
    yaw_pid = SimplePID(kp=0.8, ki=0.0, kd=0.05, min_output=10, max_output=30, deadband=4.0)

    marker_length = 15.0  # cm
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

    # 強化遠距離 (150cm+) 與移動模糊下的 Marker 偵測率
    parameters = aruco.DetectorParameters_create()
    parameters.adaptiveThreshWinSizeMin = 3
    parameters.adaptiveThreshWinSizeMax = 23
    parameters.adaptiveThreshWinSizeStep = 5
    parameters.minMarkerPerimeterRate = 0.015
    # 次像素角點精修：大幅降低距離與 yaw 的跳動，提升 PID 穩定度
    parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX

    drone = Tello()
    drone.connect()
    print("=" * 50)
    print(f"[SYSTEM] Tello Connected | Battery: {drone.get_battery()}%")
    print("快捷鍵: [1]起飛 [2]降落 [3]急停 [r]重置狀態0 [n]跳下一狀態 [b]回上一狀態")
    print("=" * 50)

    drone.streamon()
    time.sleep(2)
    frame_read = drone.get_frame_read()

    target_reached_counter = 0
    lost_marker_counter = 0

    try:
        while True:
            """============ 影像擷取與前處理 (Sense) ============"""
            frame = frame_read.frame
            if frame is None:
                cv2.waitKey(1)
                continue

            display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(display_frame, cv2.COLOR_BGR2GRAY)

            corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
            if ids is not None:
                aruco.drawDetectedMarkers(display_frame, corners, ids)

            """================================================"""
            """============ 決策與狀態機運算 (Plan) ============"""
            lr_speed, fwd_speed, up_speed, yaw_speed = 0, 0, 0, 0  # 速度重置歸零：每一幀一開始先把 lr_speed, fwd_speed, up_speed, yaw_speed 預設為 0
            status_text = "Waiting..."
            

            # ==========================================================
            # 狀態機邏輯與多重 Marker 優先權處理
            # 若無人機正在執行鑽桌子或側移等固定開迴路動作（is_executing_maneuver == True），主迴圈就不做任何干涉
            # 只有在正常狀態下，才會根據目前的 current_state（0～10）尋找對應的 Marker、計算空間誤差，並透過 PID 賦予四軸速度
            # ==========================================================
            if not is_executing_maneuver:

                # --- [第一段] State 0 & 1: 鎖定並逼近 ID 1 (50cm) ---
                if current_state in [0, 1]:
                    found, rvec, tvec, x_err, y_err, dist, yaw_deg = get_marker_pose(           # 從相機看到的所有標籤中，單獨過濾出 ID 1
                        corners, ids, 1, marker_length, intrinsic, distortion
                    )
                    if found:
                        if current_state == 0:                                                  # 若原本是 State 0，看到 ID 1 的瞬間立即提升為 State 1，並將計數器歸零
                            current_state = 1
                            target_reached_counter = 0
                            print("[STATE] 偵測到 ID 1 -> 進入 STATE 1 (逼近至 50cm)")

                        aruco.drawAxis(display_frame, intrinsic, distortion, rvec, tvec, 10)
                        z_err = dist - ID1_TARGET_DIST                                          # 誤差計算：前後目標是 50cm，因此深度誤差為 z_err = dist - 50.0
                        lr_speed = x_pid.update(x_err)                                          # PID 四軸聯動：將 4 個誤差丟給對應的 PID
                        up_speed = y_pid.update(y_err)                                          # 算出此幀的左右平移速度（lr_speed）、升降速度（up_speed）、前進速度（fwd_speed）與自轉速度（yaw_speed）
                        fwd_speed = z_pid.update(z_err)
                        yaw_speed = yaw_pid.update(yaw_deg)

                        z_tolerance = 7.5
                        x_tolerance = 15.0
                        is_in_range = (abs(z_err) <= z_tolerance) and (abs(x_err) <= x_tolerance)              # 前後距離在 50 ± 7.5 cm 內，左右偏差在 ± 15.0 cm 內（必須居中正對，不能太偏）
                        # 若某幀因為風吹或視角反光稍微出界：只減 1 而不是直接歸零，防止無人機在門檻邊緣頻繁重置而卡關
                        target_reached_counter = target_reached_counter + 1 if is_in_range else max(0, target_reached_counter - 1)
                        status_text = f"S1[ID 1] Dist:{dist:.1f}/{ID1_TARGET_DIST}cm ({target_reached_counter}/4)"

                        if is_flying and target_reached_counter >= 4:
                            target_reached_counter = 0
                            current_state = 2

                            # 動態計算實際需要的右移距離：
                            # 若 x_err = -10 (偏左)，實際需要右移 90 - (-10) = 100 cm
                            # 若 x_err = +10 (偏右)，實際需要右移 90 - (+10) = 80 cm
                            actual_right_dist = int(round(ID1_BYPASS_RIGHT - x_err))
                            # 限制 actual_right_dist 在安全範圍 (避免極端雜訊造成位移過大或過小，例如限制 ID1_BYPASS_RIGHT ± 20 cm)
                            min_safe = max(20, int(round(ID1_BYPASS_RIGHT - x_tolerance)))                                                # 20cm 為 Tello SDK 的最低物理位移限制
                            max_safe = int(round(ID1_BYPASS_RIGHT + x_tolerance))

                            actual_right_dist = max(min_safe, min(max_safe, actual_right_dist))
                            
                            print(f"[STATE] ID 1 到位！最後偏差 x_err={x_err:.1f}cm | 動態向右避障 {actual_right_dist}cm")
                            is_executing_maneuver = True  # 先鎖住主迴圈，避免 thread 啟動前又送出 RC 指令干擾 move_xxx
                            threading.Thread(
                                target=execute_maneuver,
                                args=(drone, [("right", actual_right_dist)], 3),                                            # <-- 這裡的 3 就是 next_state, i.e. current_state = 3
                                daemon=True
                            ).start()
                    else:
                        target_reached_counter = max(0, target_reached_counter - 1)
                        status_text = "S1: Searching ID 1..."

                # --- [第一段] State 3: 逼近 ID 2 (115cm) ---
                elif current_state == 3:
                    found, rvec, tvec, x_err, y_err, dist, yaw_deg = get_marker_pose(
                        corners, ids, 2, marker_length, intrinsic, distortion
                    )
                    if found:
                        aruco.drawAxis(display_frame, intrinsic, distortion, rvec, tvec, 10)
                        z_err = dist - ID2_TARGET_DIST
                        lr_speed = x_pid.update(x_err)
                        up_speed = y_pid.update(y_err)
                        fwd_speed = z_pid.update(z_err)
                        yaw_speed = yaw_pid.update(yaw_deg)

                        x_tolerance = 15
                        z_tolerance = 10.0  # 深度容許公差 ±10cm (範圍 105 ~ 125 cm)
                        is_in_range = (abs(z_err) <= z_tolerance) and (abs(x_err) <= x_tolerance)                                       # 判定條件：前後在 115 ± 10cm 內，且左右在 ±15cm 內
                        target_reached_counter = target_reached_counter + 1 if is_in_range else max(0, target_reached_counter - 1)
                        status_text = f"S3[ID 2] Dist:{dist:.1f}/{ID2_TARGET_DIST}cm ({target_reached_counter}/4)"

                        if is_flying and target_reached_counter >= 4:
                            target_reached_counter = 0
                            # 加上 reset，防止即將進入 State 4/5 時殘留累積的誤差/積分動量
                            x_pid.reset()
                            y_pid.reset()
                            z_pid.reset()
                            yaw_pid.reset()
                            # 動態計算實際需要的左移距離：
                            actual_left_dist = int(round(ID2_INIT_LEFT + x_err))
                            min_safe = max(20, int(round(ID2_INIT_LEFT - x_tolerance)))
                            max_safe = int(round(ID2_INIT_LEFT + x_tolerance))
                            actual_left_dist = max(min_safe, min(max_safe, actual_left_dist))

                            print(f"[STATE] ID 2 到位！最後偏差 x_err={x_err:.1f}cm，動態向左脫離板子 {actual_left_dist}cm")
                            is_executing_maneuver = True  # 先鎖住主迴圈，避免 thread 啟動前又送出 RC 指令干擾 move_xxx
                            threading.Thread(
                                target=execute_maneuver,
                                args=(drone, [("left", actual_left_dist)], 4),
                                daemon=True
                            ).start()
                    else:
                        # 若剛右移完還沒看到 ID 2，微幅往前推進
                        fwd_speed = 18
                        status_text = "S3: Advancing to find ID 2..."

                # --- [第一段轉第二段] State 4: 持續往左平移直到看見手持 ID 0 ---
                elif current_state == 4:
                    found_0, _, _, _, _, dist_0, _ = get_marker_pose(
                        corners, ids, 0, marker_length, intrinsic, distortion
                    )
                    if found_0:
                        print(f"[STATE] 捕獲手持 ID 0 (距離 {dist_0:.1f}cm)！進入 STATE 5 跟隨模式")
                        x_pid.reset()
                        y_pid.reset()
                        z_pid.reset()
                        yaw_pid.reset()
                        current_state = 5
                    else:
                        lr_speed = SCAN_ID0_LEFT_SPEED
                        status_text = "S4: Sliding Left to find Handheld ID 0..."

                # --- [第二段] State 5: 跟隨手持 ID 0 繞椅子，並自動偵測 ID 3 交接 ---
                elif current_state == 5:
                    # 優先檢查是否已經繞完椅子看到桌上的 ID 3 (免遮擋自動交接)
                    found_3, _, _, _, _, dist_3, _ = get_marker_pose(
                        corners, ids, 3, marker_length, intrinsic, distortion
                    )
                    if found_3 and dist_3 <= ID3_HANDOFF_DIST:
                        print(f"[STATE] 偵測到桌前 ID 3 (距離 {dist_3:.1f}cm)！自動脫離 ID 0 進入 STATE 6")
                        target_reached_counter = 0
                        x_pid.reset()
                        y_pid.reset()
                        z_pid.reset()
                        yaw_pid.reset()
                        current_state = 6
                        continue

                    found_0, rvec, tvec, x_err, y_err, dist_0, yaw_deg = get_marker_pose(
                        corners, ids, 0, marker_length, intrinsic, distortion
                    )
                    if found_0:
                        lost_marker_counter = 0
                        aruco.drawAxis(display_frame, intrinsic, distortion, rvec, tvec, 10)
                        z_err = dist_0 - ID0_FOLLOW_DIST

                        # 跟隨手持 Marker 時放大推力上限，讓無人機能緊跟人的步伐與轉彎
                        lr_speed = x_pid.update(x_err, custom_max=35)
                        up_speed = y_pid.update(y_err, custom_max=28)
                        fwd_speed = z_pid.update(z_err, custom_max=38)
                        # 手持 marker 角度不固定，改為追蹤 marker 所在方位角 (bearing)，繞椅子更順暢
                        # marker 在右 (x_err > 0) → 誤差為負 → 送出 -yaw_speed 為正 → 右轉
                        yaw_follow_err = -math.degrees(math.atan2(x_err, dist_0))
                        yaw_speed = yaw_pid.update(yaw_follow_err, custom_max=35)

                        status_text = f"S5[Follow ID 0] Dist:{dist_0:.1f}cm (Looking for ID 3)"
                    else:
                        lost_marker_counter += 1
                        status_text = f"S5: ID 0 Lost ({lost_marker_counter}) - Hovering..."

                # --- [第三段] State 6: 對準桌緣 ID 3 準備鑽桌底 ---
                elif current_state == 6:
                    found_3, rvec, tvec, x_err, y_err, dist_3, yaw_deg = get_marker_pose(
                        corners, ids, 3, marker_length, intrinsic, distortion
                    )
                    if found_3:
                        aruco.drawAxis(display_frame, intrinsic, distortion, rvec, tvec, 10)
                        z_err = dist_3 - ID3_TARGET_DIST
                        lr_speed = x_pid.update(x_err)
                        up_speed = y_pid.update(y_err)
                        fwd_speed = z_pid.update(z_err)
                        yaw_speed = yaw_pid.update(yaw_deg)

                        # 參數設定的重點是 ID3_TARGET_DIST, x_tolerance, z_tolerance, yaw_tolerance 必須彼此搭配，確保可以正對到 ID 3
                        z_tolerance = 15.0   # 深度容許範圍
                        x_tolerance = 15.0   # 水平對齊必須嚴格 (± 8cm 確保進洞居中)
                        yaw_tolerance = 15 # 航向必須正對 (± 8度 避免盲飛撞桌腳)
                        ## 原本180-yaw_deg的寫法會讓tolerance不成立
                        yaw_diff = yaw_deg  # get_marker_pose 已修正為正對 ≈ 0°

                        is_in_range = (abs(z_err) <= z_tolerance) and (abs(x_err) <= x_tolerance) and (abs(yaw_diff) <= yaw_tolerance)
                        target_reached_counter = target_reached_counter + 1 if is_in_range else max(0, target_reached_counter - 1)
                        status_text = f"S6[ID 3 Table] Dist:{dist_3:.1f}/{ID3_TARGET_DIST}cm ({target_reached_counter}/15)"

                        if is_flying and target_reached_counter >= 15:
                            target_reached_counter = 0
                            current_state = 7
                            # 鑽桌前重置 PID，消除殘存動量與前次誤差
                            x_pid.reset()
                            y_pid.reset()
                            z_pid.reset()
                            yaw_pid.reset()

                            # 當 dist_3 = 90 (停得較遠)，z_err = +10 -> 前進 240 + 10 = 250
                            # 當 dist_3 = 70 (停得較近)，z_err = -10 -> 前進 240 - 10 = 230
                            actual_fwd_dist = int(round(TABLE_FWD_DIST + z_err))

                            print("[STATE] ID 3 對準完成！執行鑽桌子與旋轉爬升動作")
                            table_actions = [
                                ("down", TABLE_DOWN_DIST),
                                ("forward", actual_fwd_dist),
                                ("cw", TABLE_CW_DEG),
                                ("back", TABLE_BACK_DIST),
                                ("up", TABLE_UP_DIST)
                            ]
                            is_executing_maneuver = True  # 先鎖住主迴圈，避免 thread 啟動前又送出 RC 指令干擾 move_xxx
                            threading.Thread(
                                target=execute_maneuver,
                                args=(drone, table_actions, 8),
                                daemon=True
                            ).start()
                    else:
                        target_reached_counter = max(0, target_reached_counter - 1)
                        status_text = "S6: Searching ID 3..."

                # --- [第三段轉第四段] State 8: 定位牆上 ID 4 (保持 145cm) 並拉高準備越過板子 ---
                elif current_state == 8:
                    found_4, rvec, tvec, x_err, y_err, dist_4, yaw_deg = get_marker_pose(
                        corners, ids, 4, marker_length, intrinsic, distortion
                    )
                    if found_4:
                        aruco.drawAxis(display_frame, intrinsic, distortion, rvec, tvec, 10)
                        z_err = dist_4 - ID4_TARGET_DIST
                        lr_speed = x_pid.update(x_err)
                        up_speed = y_pid.update(y_err)
                        fwd_speed = z_pid.update(z_err)
                        yaw_speed = yaw_pid.update(yaw_deg)

                        z_tolerance = 15.0
                        x_tolerance = 15.0
                        yaw_tolerance = 15.0

                        ## 原本180-yaw_deg的寫法會讓tolerance不成立
                        yaw_diff = yaw_deg  # get_marker_pose 已修正為正對 ≈ 0°

                        is_in_range = (
                            (abs(z_err) <= z_tolerance) and 
                            (abs(x_err) <= x_tolerance) and 
                            (abs(yaw_diff) <= yaw_tolerance)
                        )
                        target_reached_counter = target_reached_counter + 1 if is_in_range else max(0, target_reached_counter - 1)
                        status_text = f"S8[ID 4 Wall] Dist:{dist_4:.1f}/{ID4_TARGET_DIST}cm ({target_reached_counter}/15)"

                        if is_flying and target_reached_counter >= 15:
                            target_reached_counter = 0
                            # 清空 PID，避免前次推進動量帶入下一個動作
                            x_pid.reset()
                            y_pid.reset()
                            z_pid.reset()
                            yaw_pid.reset()

                            print("[STATE] ID 4 定位完成！微幅向後並開始向左飛越板子上方")
                            # 為了防止找不到 ID 5 BACK_FROM_ID4_DIST (可設 20~30cm)，再進入 State 9 平移
                            is_executing_maneuver = True  # 先鎖住主迴圈，避免 thread 啟動前又送出 RC 指令干擾 move_xxx
                            threading.Thread(
                                target=execute_maneuver,
                                args=(drone, [("back", BACK_FROM_ID4_DIST), 
                                              ("up", OVER_BOARD_UP_DIST)], 9),
                                daemon=True
                            ).start()
                    else:
                        target_reached_counter = max(0, target_reached_counter - 1)
                        # 剛從桌底鑽出來若高度不夠沒看到 ID 4，緩慢上升並退後擴大視野
                        up_speed = SCAN_ID4_UP_SPEED
                        fwd_speed = SCAN_ID4_BACK_SPEED
                        status_text = "S8: Climbing/Backing to find ID 4 on Wall..."

                # --- [第四段] State 9: 持續向左水平移動越過板子，直到看見 ID 5 ---
                elif current_state == 9:
                    found_5, _, _, _, _, dist_5, _ = get_marker_pose(
                        corners, ids, 5, marker_length, intrinsic, distortion
                    )
                    if found_5:
                        print(f"[STATE] 捕獲終點 ID 5 (距離 {dist_5:.1f}cm)！進入 STATE 10 對準降落區")
                        target_reached_counter = 0
                        x_pid.reset()
                        y_pid.reset()
                        z_pid.reset()
                        yaw_pid.reset()
                        current_state = 10
                    else:
                        lr_speed = SCAN_ID5_LEFT_SPEED
                        status_text = "S9: Flying Left over Board to find ID 5..."

                # --- [第五段] State 10: 對準 ID 5 並後退至 150cm 降落於桌上 ---
                elif current_state == 10:
                    found_5, rvec, tvec, x_err, y_err, dist_5, yaw_deg = get_marker_pose(
                        corners, ids, 5, marker_length, intrinsic, distortion
                    )
                    if found_5:
                        aruco.drawAxis(display_frame, intrinsic, distortion, rvec, tvec, 10)
                        z_err = dist_5 - ID5_TARGET_DIST
                        lr_speed = x_pid.update(x_err)
                        up_speed = y_pid.update(y_err)
                        fwd_speed = z_pid.update(z_err)
                        yaw_speed = yaw_pid.update(yaw_deg)

                        z_tolerance = 15.0
                        x_tolerance = 15.0
                        yaw_tolerance = 15.0

                        ## 原本180-yaw_deg的寫法會讓tolerance不成立
                        yaw_diff = yaw_deg  # get_marker_pose 已修正為正對 ≈ 0°

                        is_in_range = (
                            (abs(z_err) <= z_tolerance) and 
                            (abs(x_err) <= x_tolerance) and 
                            (abs(yaw_diff) <= yaw_tolerance)
                        )
                        target_reached_counter = target_reached_counter + 1 if is_in_range else max(0, target_reached_counter - 1)
                        status_text = f"S10[ID 5 Land] Dist:{dist_5:.1f}/{ID5_TARGET_DIST}cm ({target_reached_counter}/23)"

                        if is_flying and target_reached_counter >= 23:
                            target_reached_counter = 0
                            current_state = 11  # 立即切換，防止下一幀又重複觸發
                            
                            # 消除 PID 殘餘動量
                            x_pid.reset()
                            y_pid.reset()
                            z_pid.reset()
                            yaw_pid.reset()

                            # 動態計算實際後退距離：當前距離為 dist_5，目標到達 DESIRED_LAND_DIST (150cm)
                            ID5_BACK_BASE = int(round(DESIRED_LAND_DIST - ID5_TARGET_DIST))  # 基準量 100 cm
                            actual_back_dist = int(round(ID5_BACK_BASE - z_err))

                            # 限制 actual_back_dist 在安全範圍 (避免極端雜訊造成位移過大或過小，限制基準值 ± z_tolerance)
                            min_safe = max(20, int(round(ID5_BACK_BASE - z_tolerance)))      # 20cm 為 Tello SDK 的最低物理位移限制
                            max_safe = int(round(ID5_BACK_BASE + z_tolerance))

                            actual_back_dist = max(min_safe, min(max_safe, actual_back_dist))

                            print(f"[STATE] ID 5 對準完成！當前距離 {dist_5:.1f}cm | 動態後退 {actual_back_dist}cm 至桌面中央降落")

                            def land_sequence(back_dist):
                                # 1. 執行精確補償後退
                                execute_maneuver(drone, [("back", back_dist)], 11)  # 明確指定 11，避免 current_state += 1 變成 12
                                # 2. 後退完成，停懸 0.5 秒消除煞車慣性與地面氣流擾動
                                time.sleep(0.5)
                                # 3. 垂直落桌 (單次觸發)
                                async_land(drone)

                            is_executing_maneuver = True
                            threading.Thread(target=land_sequence, args=(actual_back_dist,), daemon=True).start()
                    else:
                        target_reached_counter = max(0, target_reached_counter - 1)
                        status_text = "S10: Searching ID 5 for Landing..."

                elif current_state == 11:
                    status_text = "S11: Track Complete! Landed."

            # ========================================================================================
            #                                 HUD 畫面疊加資訊配置圖
            # ========================================================================================
            # 視窗左上角 (0, 0)
            # +------------------------------------------------------------------------------------+
            # |                                                                                    |
            # |  (20, 30)  [黃字] State: <0~11> | Maneuver: <True/False>                           |
            # |  (20, 65)  [綠字] <status_text: 當前目標 Marker / 距離 / 達標計數 (3/4) / 搜尋狀態>  |
            # |  (20, 100) [青字] RC: [LR:<左右>, FB:<前後>, UD:<上下>, YAW:<旋轉>]                |
            # |                                                                                    |
            # |                                                                                    |
            # |                                                                                    |
            # |                           [綠框: 偵測到的 Marker 角點]                             |
            # |                             +-------------+                                        |
            # |                             |  ID: <0~5>  |                                        |
            # |                             |    (X,Y,Z)  | <--- 3D 姿態軸 (紅=X, 綠=Y, 藍=Z)      |
            # |                             +-------------+                                        |
            # |                                                                                    |
            # +------------------------------------------------------------------------------------+
            # 【圖層說明】
            # 1. 畫面底層：Tello 鏡頭即時串流畫面 (display_frame)
            # 2. 視覺標註：aruco.drawDetectedMarkers (綠色方框與ID) + aruco.drawAxis (3D坐標軸)
            # 3. 抬頭文字：3 行 cv2.putText，即時監控無人機的狀態機、對準進度與四軸輸出速度
            # ========================================================================================
            cv2.putText(display_frame, f"State: {current_state} | Maneuver: {is_executing_maneuver}",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 200, 0), 2)
            cv2.putText(display_frame, status_text,
                        (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.putText(display_frame, f"RC: [LR:{lr_speed}, FB:{fwd_speed}, UD:{up_speed}, YAW:{-yaw_speed}]",
                        (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.imshow("Midterm Track - EAB06 Full Autonomous", display_frame)

            # ==========================================================
            # 鍵盤監聽與 RC 指令發送
            # ==========================================================
            key = cv2.waitKey(1) & 0xFF
            if key != 255 and key != -1:
                if key == 27:  # ESC 緊急降落並退出
                    threading.Thread(target=async_land, args=(drone,), daemon=True).start()
                    time.sleep(1)
                    break
                keyboard(drone, key)
            else:
                # 只要無人機正在飛、且沒有在做 Maneuver、且尚未抵達終點降落（current_state < 11），就呼叫 drone.send_rc_control(...) 將算好的 PID 速度送給無人機
                if is_flying and not is_executing_maneuver and current_state < 11:
                    drone.send_rc_control(lr_speed, fwd_speed, up_speed, -yaw_speed)

    except KeyboardInterrupt:
        drone.land()
    finally:
        drone.send_rc_control(0, 0, 0, 0)
        drone.streamoff()
        cv2.destroyAllWindows()

    try:
        print(f"[SYSTEM] Remaining Battery: {drone.get_battery()}%")
    except Exception as e:
        print(f"[SYSTEM] Could not read battery: {e}")

if __name__ == "__main__":
    main()