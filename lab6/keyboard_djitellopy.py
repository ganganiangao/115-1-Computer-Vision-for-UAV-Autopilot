# keyboard_djitellopy.py

def keyboard(drone, key):
    print("key:", key)
    fb_speed = 40  # 前後速度
    lf_speed = 40  # 左右速度
    ud_speed = 50  # 上下速度
    degree = 30    # 旋轉速度 / 偏航

    # 1: 起飛
    if key == ord('1'):
        drone.takeoff()

    # 2: 降落
    elif key == ord('2'):
        drone.land()

    # 3: 緊急停止所有運動 (速度皆設為 0)
    elif key == ord('3'):
        drone.send_rc_control(0, 0, 0, 0)
        print("stop!!!!")

    # 前後控制 (forward / backward)
    elif key == ord('w'):
        drone.send_rc_control(0, fb_speed, 0, 0)
        print("forward!!!!")
    elif key == ord('s'):
        drone.send_rc_control(0, (-1) * fb_speed, 0, 0)
        print("backward!!!!")

    # 左右平移 (left / right)
    elif key == ord('a'):
        drone.send_rc_control((-1) * lf_speed, 0, 0, 0)
        print("left!!!!")
    elif key == ord('d'):
        drone.send_rc_control(lf_speed, 0, 0, 0)
        print("right!!!!")

    # 上下升降 (up / down) - 注意投影片中 x 為 up, z 為 down
    elif key == ord('x'):
        drone.send_rc_control(0, 0, ud_speed, 0)
        print("up!!!!")
    elif key == ord('z'):
        drone.send_rc_control(0, 0, (-1) * ud_speed, 0)
        print("down!!!!")

    # 旋轉偏航 (rotate / counter rotate)
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