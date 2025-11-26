import mss
import numpy as np

class Capture:
    def __init__(self, area):
        self._mss = mss.mss()
        self.area = area
    
    
    def get_frame(self):
        img = self._mss.grab(self.area)
        bgra_img_array = np.asarray(img)
        bgr_img_array = bgra_img_array[..., :3]
        return bgr_img_array
