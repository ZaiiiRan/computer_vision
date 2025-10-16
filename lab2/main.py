import cv2
import numpy as np

cap = cv2.VideoCapture(0)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv, np.array([0, 150, 120]), np.array([20, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([160, 150, 120]), np.array([180, 255, 255]))
    mask = mask1 | mask2

    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)

    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.erode(mask, kernel, iterations=1)

    moments = cv2.moments(mask)
    area = moments['m00'] / 255

    frame = cv2.bitwise_and(frame, frame, mask=mask)
    cv2.putText(frame, f"Area: {area} px", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Red tracker", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
