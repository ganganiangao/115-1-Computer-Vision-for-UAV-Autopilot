# lab6.py
import cv2
from djitellopy import Tello
from keyboard_djitellopy import keyboard

# 初始化並連線無人機
drone = Tello()
drone.connect()
drone.streamon()  # 開啟前鏡頭影像串流
frame_read = drone.get_frame_read()

while True:
    frame = frame_read.frame
    cv2.imshow("Tello Stream", frame)

    # 讀取按鍵輸入
    key = cv2.waitKey(1) & 0xFF

    # 投影片指示：if key != -1 優先執行手動控制，確保最高優先權
    if key != 255 and key != -1:
        if key == 27:  # 按 ESC 鍵結束
            drone.land()
            break
        keyboard(drone, key)
    else:
        # 沒有任何按鍵時，才在此處執行自動飛行 / Marker 追蹤控制
        pass

drone.streamoff()
cv2.destroyAllWindows()