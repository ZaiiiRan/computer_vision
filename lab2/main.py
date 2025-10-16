import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([20, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([160, 100, 100]), np.array([180, 255, 255]))
    mask = mask1 | mask2

    cv2.imshow("Red tracker", cv2.bitwise_and(frame, frame, mask=mask))

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
