import pyautogui
import numpy as np

class Aimbot:
    def __init__(self, area, sensitivity=0.9, lock_distance=100):
        self.center_x = area["width"] // 2
        self.center_y = area["height"] // 2

        self.offset_x = area["left"]
        self.offset_y = area["top"]

        self.sensitivity = sensitivity
        self.lock = False
        self.lock_distance = lock_distance


    def process(self, detections):
        if not detections:
            self.lock = False
            return False

        target, distance = self._select_closest(detections)

        if not target:
            return False

        self.lock = distance < self.lock_distance
        tx, ty = self._aim_point(target["box"])
        self._move_cursor(tx, ty)

        if self.lock:
            pyautogui.mouseDown()
            pyautogui.mouseUp()
        return distance < self.lock_distance


    def _select_closest(self, detections):
        best = None
        best_distance = float("inf")

        for det in detections:
            box = det["box"]
            cx = (box[0] + box[2]) // 2
            cy = (box[1] + box[3]) // 2

            distance = np.hypot(cx - self.center_x, cy - self.center_y)

            if distance < best_distance:
                best_distance = distance
                best = det

        return best, best_distance


    def _aim_point(self, box):
        x1, y1, x2, y2 = box
        center_x = (x1 + x2) // 2
        center_y = y1 + (y2 - y1) // 4
        return center_x, center_y


    def _move_cursor(self, lx, ly):
        gx = lx + self.offset_x
        gy = ly + self.offset_y

        cur_x, cur_y = pyautogui.position()

        dx = (gx - cur_x) * self.sensitivity
        dy = (gy - cur_y) * self.sensitivity

        if abs(dx) > 1 or abs(dy) > 1:
            pyautogui.moveRel(int(dx), int(dy), duration=0.005)
