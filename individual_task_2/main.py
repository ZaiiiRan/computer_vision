import pyautogui
import time
import cv2
from capture import Capture
from yolo import YoloModel
from aimbot import Aimbot

DELAY = 0.01


def vizualize_detections(frame, detections, aimbot):
    if frame is None or frame.size == 0:
        return

    display_frame = frame.copy()

    for det in detections:
        box = det['box']
        conf = det['conf']

        cv2.rectangle(display_frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 3)

        cv2.putText(display_frame, f"{conf:.2f}", (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 3)
        
        aim_x, aim_y = aimbot._aim_point(box)
        cv2.circle(display_frame, (aim_x, aim_y), 5, (0, 0, 255), -1)
    cv2.circle(display_frame, (aimbot.center_x, aimbot.center_y), 3, (0, 0, 255), -1)

    cv2.imshow("Detections", display_frame)
    cv2.waitKey(1)


def main():
    pyautogui.FAILSAFE = False
    pyautogui.PAUSE = 0.001

    width, height = pyautogui.size()
    monitor_area = {
        'top': 0,
        'left': 0,
        'width': width,
        'height': height
    }
    
    model_path = './yolo_aimbot6/weights/best.pt'

    cap = Capture(monitor_area)
    model = YoloModel(model_path, device='cpu')
    aimbot = Aimbot(monitor_area, sensitivity=0.9, lock_distance=100)

    while True:
        start_time = time.time()
        frame = cap.get_frame()
        detections = model.predict(frame, conf_threshold=0.75)
        aimbot.process(detections)
        vizualize_detections(frame, detections, aimbot)
        end_time = time.time()
        elapsed_time = end_time - start_time
        sleep_time = DELAY - elapsed_time

        if sleep_time > 0:
            time.sleep(sleep_time)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
