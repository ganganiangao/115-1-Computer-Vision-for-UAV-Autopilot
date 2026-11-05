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

# =====================================================================
# [擴充] 第二 ~ 五段 (STATE >= 6)
# 以下為新增程式碼，STATE 0~5 (第一段) 的既有邏輯完全不動。
# =====================================================================
EXT_FIRST_STATE = 6
STATE_FOLLOW_0 = 6      # 第二段：跟隨手持 marker 0，直到看到 marker 3
STATE_SEG3_MOVE = 7     # 第三段(a)：往下 -> 水平移動 -> 向右旋轉 (離散動作)
STATE_SEARCH_LEFT = 8   # 第三段(b)：穿過桌子、上升、右轉後，短距離向左平移搜尋 marker
STATE_STRAFE_5 = 9      # 第四段：向左水平移動直到看到 marker 5
STATE_SEG5_MOVE = 10    # 第五段：往後 120cm -> 緩慢降落
STATE_END = 11          # 全部完成

# ---- 可調參數 (請依現場微調) ----
EXT_HOLD_FRAMES = 3         # 連續幾幀看到 marker 才確認 (軟性計數，掉幀只扣 1)
FOLLOW_DIST_0 = 70.0        # 第二段：與手持 marker 0 保持的距離 (cm)，越近越容易穩定偵測
FOLLOW_LOST_SEC = 2.0       # 第二段：marker 0 暫時跑出畫面時，朝最後看到的方向繼續轉多久 (秒)
FOLLOW_LOST_YAW = 20        # 上述搜尋轉向的速度
SEG3_TRIGGER_DIST = 200.0   # 第二段：看到 marker 3 且距離 <= 此值才進入第三段 (cm)
SEG3_ACTIONS = [            # 第三段：往下 -> 穿過桌子 -> 往上 -> 向右旋轉
    ("down", 40),           #   下降量 (cm)，Tello 最小 20，請依桌高調整
    ("forward", 65),        #   穿過桌子的水平距離 (cm)
    ("up", 40),             #   穿過後上升 (cm)，通常與下降量相同以回到原高度
    ("cw", 90),             #   向右(順時針)旋轉角度
]
SEARCH_ID = 4               # 右轉後向左平移要搜尋的 marker id
SEARCH_LEFT_SPEED = 15      # 搜尋時向左平移速度 (慢一點，避免平移太多)
SEARCH_UP_SPEED = 0         # 搜尋時順便上升的速度 (marker 偏高、看不到時可設 10~15)
SEARCH_TIMEOUT = 6.0        # 搜尋最長時間 (秒)，超時懸停並暫停自動流程
STRAFE_LEFT_SPEED = 20      # 第四段：向左水平速度
STRAFE_TIMEOUT = 20.0       # 向左最長時間 (秒)，超時懸停並暫停自動流程
SEG5_ACTIONS = [("back", 120)]   # 第五段：往後水平移動 (cm)
SLOW_DESCEND_SPEED = 15     # 緩慢降落速度
LAND_CUTOFF_CM = 20         # 高度(雷射測距)低於此值就呼叫 land()
SLOW_LAND_TIMEOUT = 15.0    # 緩降最長時間 (秒)
SETTLE_SEC = 1.0            # 離散動作開始前先懸停的時間，等 rc 殘餘速度消掉 (避免指令被拒絕)
MANEUVER_RETRIES = 1        # 離散指令被 Tello 拒絕時的重試次數
DESCEND_RC_SPEED = 25       # 下降指令失敗時，改用 rc + 雷射測距高度閉迴路下降的速度
RC_DESCEND_TIMEOUT = 6.0    # 閉迴路下降最長時間 (秒)

# 擴充用狀態 (用 dict 存放，避免到處宣告 global)
EXT = {"cnt": 0, "t0": None, "paused": False, "last_ang": 0.0, "last_seen": None}

# 第二段跟隨 marker 0 用的 PID (沿用既有 SimplePID)
f_fwd_pid = SimplePID(kp=0.5, ki=0.0, kd=0.03, min_output=12, max_output=35)
f_up_pid = SimplePID(kp=0.5, ki=0.0, kd=0.05, min_output=15, max_output=30)
f_yaw_pid = SimplePID(kp=1.0, ki=0.0, kd=0.04, min_output=12, max_output=45)   # 輸入為水平夾角 (度)


def go(new_state):
    """切換擴充狀態並重置計數器"""
    global current_state
    current_state = new_state
    EXT["cnt"] = 0
    EXT["t0"] = None
    EXT["last_seen"] = None
    print(f"[STATE] Successfully transitioned to STATE {current_state}")


def get_marker_poses(corners, ids, marker_length, intrinsic, distortion):
    """一次估計所有看到的 marker 姿態，回傳 {id: (rvec, tvec)}。之後依 id 取用，
    因此同時看到多個 marker 時不會搞混。"""
    poses = {}
    if ids is None or len(corners) == 0:
        return poses
    rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_length, intrinsic, distortion)
    for i, mid in enumerate(ids.flatten()):
        poses[int(mid)] = (rvecs[i][0], tvecs[i][0])
    return poses


def slow_descend_and_land(drone):
    """以固定慢速下降，直到雷射測距高度低於門檻再降落"""
    global is_flying
    t0 = time.time()
    while is_flying and time.time() - t0 < SLOW_LAND_TIMEOUT:
        try:
            h = drone.get_height()
        except Exception:
            h = None
        if h is not None and 0 < h <= LAND_CUTOFF_CM:
            break
        drone.send_rc_control(0, 0, -SLOW_DESCEND_SPEED, 0)
        time.sleep(0.2)
    drone.send_rc_control(0, 0, 0, 0)
    try:
        drone.land()
    except Exception as e:
        print(f"[Land Error] {e}")
    is_flying = False
    print("[INFO] Land complete")


def _safe_height(drone):
    try:
        return drone.get_height()
    except Exception:
        return None


def rc_descend(drone, cm):
    """備援：用 rc 往下飛，以雷射測距高度閉迴路判斷，不依賴時間校正"""
    h0 = _safe_height(drone)
    if h0 is None:
        return False
    target = max(h0 - cm, LAND_CUTOFF_CM)
    t0 = time.time()
    while is_flying and time.time() - t0 < RC_DESCEND_TIMEOUT:
        h = _safe_height(drone)
        if h is not None and h <= target:
            drone.send_rc_control(0, 0, 0, 0)
            return True
        drone.send_rc_control(0, 0, -DESCEND_RC_SPEED, 0)
        time.sleep(0.15)
    drone.send_rc_control(0, 0, 0, 0)
    return False


def run_action(drone, action, val):
    """執行單一離散動作，檢查 Tello 回應 (djitellopy 被拒絕時回傳 False 而不是丟例外)。"""
    fns = {
        "right": drone.move_right, "left": drone.move_left,
        "forward": drone.move_forward, "back": drone.move_back,
        "down": drone.move_down, "up": drone.move_up,
        "cw": drone.rotate_clockwise, "ccw": drone.rotate_counter_clockwise,
    }
    fn = fns[action]
    for attempt in range(MANEUVER_RETRIES + 1):
        try:
            ok = fn(val)
        except Exception as e:
            print(f"[MANEUVER ERROR] {action} {val}: {e}")
            ok = False
        if ok is not False:
            return True
        print(f"[MANEUVER FAIL] {action} {val} rejected (try {attempt + 1}/{MANEUVER_RETRIES + 1}), "
              f"height={_safe_height(drone)}")
        drone.send_rc_control(0, 0, 0, 0)
        time.sleep(0.8)
    if action == "down":
        print("[MANEUVER] SDK down failed, fallback to closed-loop rc descent")
        return rc_descend(drone, val)
    return False


def execute_ext_maneuver(drone, actions, next_state, land_at_end=False):
    """擴充版離散動作 (支援 down/up/cw/ccw)。呼叫端須先把 is_executing_maneuver 設成 True。
    任何一步失敗就中止整段並懸停暫停，避免例如「下降失敗卻繼續往上飛」頂到桌底。"""
    global is_executing_maneuver
    drone.send_rc_control(0, 0, 0, 0)
    time.sleep(SETTLE_SEC)
    failed = False
    for action, val in actions:
        if not is_flying:
            break
        print(f"[MANEUVER] Executing {action} {val} (height={_safe_height(drone)}) ...")
        if not run_action(drone, action, val):
            print(f"[MANEUVER ABORT] {action} {val} failed. Hover and pause. "
                  f"Fly manually, then press n to resume.")
            failed = True
            break
        time.sleep(0.5)
    drone.send_rc_control(0, 0, 0, 0)
    if failed:
        EXT["paused"] = True
        is_executing_maneuver = False
        return
    if land_at_end and is_flying:
        slow_descend_and_land(drone)
    go(next_state)
    is_executing_maneuver = False


def start_ext_maneuver(drone, actions, next_state, land_at_end=False):
    """先在主執行緒把旗標設好再開 thread，避免下一幀重複觸發"""
    global is_executing_maneuver
    is_executing_maneuver = True
    threading.Thread(target=execute_ext_maneuver,
                     args=(drone, actions, next_state, land_at_end), daemon=True).start()


def run_ext_state(drone, poses):
    """第二~五段狀態機。回傳這一幀要送的 RC (left_right, fwd, up, yaw)。"""
    now = time.time()
    st = current_state
    stop = (0, 0, 0, 0)

    # ---------- 第二段：跟隨 marker 0，看到 marker 3 就進入第三段 ----------
    if st == STATE_FOLLOW_0:
        if 3 in poses and poses[3][1][2] <= SEG3_TRIGGER_DIST:
            EXT["cnt"] += 1
        else:
            EXT["cnt"] = max(0, EXT["cnt"] - 1)
        if EXT["cnt"] >= EXT_HOLD_FRAMES:
            print("[STATE] ID 3 Detected! Entering SEG 3 (down / move / rotate)")
            go(STATE_SEG3_MOVE)
            start_ext_maneuver(drone, SEG3_ACTIONS, STATE_SEARCH_LEFT)
            return stop
        if 0 in poses:
            tvec = poses[0][1]
            ang = math.degrees(math.atan2(tvec[0], tvec[2]))   # marker 相對機頭的水平夾角，右為正
            fwd = f_fwd_pid.update(tvec[2] - FOLLOW_DIST_0)
            up = f_up_pid.update(-tvec[1])
            yaw = f_yaw_pid.update(ang)                          # 以角度誤差轉向，距離遠近都一樣靈敏
            EXT["last_ang"] = ang
            EXT["last_seen"] = now
            return (0, fwd, up, yaw)
        # 看不到 marker 0：短時間內朝最後看到的方向繼續轉，把它找回來；太久就懸停
        if EXT["last_seen"] is not None and now - EXT["last_seen"] < FOLLOW_LOST_SEC:
            return (0, 0, 0, FOLLOW_LOST_YAW if EXT["last_ang"] > 0 else -FOLLOW_LOST_YAW)
        return stop

    # ---------- 第三段(a)：由按鍵跳關進來時才會走到這裡 ----------
    if st == STATE_SEG3_MOVE:
        start_ext_maneuver(drone, SEG3_ACTIONS, STATE_SEARCH_LEFT)
        return stop

    # ---------- 第三段(b)：向左短距離平移，搜尋 SEARCH_ID ----------
    if st == STATE_SEARCH_LEFT:
        if EXT["t0"] is None:
            EXT["t0"] = now
        if SEARCH_ID in poses:
            EXT["cnt"] += 1
        else:
            EXT["cnt"] = max(0, EXT["cnt"] - 1)
        if EXT["cnt"] >= EXT_HOLD_FRAMES:
            print(f"[STATE] ID {SEARCH_ID} Detected! Entering SEG 4 (strafe left to ID 5)")
            go(STATE_STRAFE_5)
            return stop
        if now - EXT["t0"] > SEARCH_TIMEOUT:
            print(f"[WARN] Search timeout, marker {SEARCH_ID} not seen. Hover and pause")
            EXT["paused"] = True
            return stop
        return (-SEARCH_LEFT_SPEED, 0, SEARCH_UP_SPEED, 0)

    # ---------- 第四段：向左水平移動直到看到 marker 5 ----------
    if st == STATE_STRAFE_5:
        if EXT["t0"] is None:
            EXT["t0"] = now
        if 5 in poses:
            EXT["cnt"] += 1
        else:
            EXT["cnt"] = max(0, EXT["cnt"] - 1)
        if EXT["cnt"] >= EXT_HOLD_FRAMES:
            print("[STATE] ID 5 Detected! Entering SEG 5 (back + slow land)")
            go(STATE_SEG5_MOVE)
            start_ext_maneuver(drone, SEG5_ACTIONS, STATE_END, land_at_end=True)
            return stop
        if now - EXT["t0"] > STRAFE_TIMEOUT:
            print("[WARN] Strafe timeout, marker 5 not seen. Hover and pause")
            EXT["paused"] = True
            return stop
        return (-STRAFE_LEFT_SPEED, 0, 0, 0)

    # ---------- 第五段：由按鍵跳關進來時才會走到這裡 ----------
    if st == STATE_SEG5_MOVE:
        start_ext_maneuver(drone, SEG5_ACTIONS, STATE_END, land_at_end=True)
        return stop

    return stop


def handle_ext_key(key):
    """擴充按鍵：3 暫停自動流程(懸停)、n 繼續；7/8/m/9/b 為測試用跳關 (m = 向左搜尋) (需已起飛)"""
    if key == ord("3"):
        EXT["paused"] = True
        print("[EXT] paused (hover)")
        return
    if key == ord("n"):
        EXT["paused"] = False
        print("[EXT] resumed")
        return
    if not is_flying or is_executing_maneuver:
        return
    if key == ord("7"):
        EXT["paused"] = False
        go(STATE_FOLLOW_0)
    elif key == ord("8"):
        EXT["paused"] = False
        go(STATE_SEG3_MOVE)
    elif key == ord("m"):
        EXT["paused"] = False
        go(STATE_SEARCH_LEFT)
    elif key == ord("9"):
        EXT["paused"] = False
        go(STATE_STRAFE_5)
    elif key == ord("b"):
        EXT["paused"] = False
        go(STATE_SEG5_MOVE)


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
                    threading.Thread(target=execute_maneuver, args=(drone, [("right", 120)]), daemon=True).start()

                elif current_state == 3 and target_reached_counter >= 4:
                    print("[STATE] Reached 115cm (within tolerance) from ID 2! Entering STATE 4 (Bypass Left 150cm)")
                    target_reached_counter = 0
                    current_state = 4
                    threading.Thread(target=execute_maneuver, args=(drone, [("left", 100)]), daemon=True).start()

                elif current_state == 5:
                    print("[STATE] Final Leg! Moving forward 125cm and stopping")
                    threading.Thread(target=execute_maneuver, args=(drone, [("forward", 50)]), daemon=True).start()

            # =====================================================
            # [擴充] 第二~五段 (STATE >= 6)；STATE 0~5 完全不受影響
            # =====================================================
            ext_rc = (0, 0, 0, 0)
            if (is_flying and not is_executing_maneuver
                    and current_state >= EXT_FIRST_STATE and not EXT["paused"]):
                ext_poses = get_marker_poses(corners, ids, marker_length, intrinsic, distortion)
                ext_rc = run_ext_state(drone, ext_poses)
                if ext_poses:
                    cv2.putText(display_frame, f"Seen IDs: {sorted(ext_poses.keys())}", (20, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if current_state >= EXT_FIRST_STATE:
                cv2.putText(display_frame, f"EXT paused: {EXT['paused']} | RC: {ext_rc}", (20, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

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
                handle_ext_key(key)
            else:
                if is_flying and not is_executing_maneuver and current_state >= EXT_FIRST_STATE:
                    drone.send_rc_control(*ext_rc)
                elif is_flying and not is_executing_maneuver:
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