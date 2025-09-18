import cv2
import time

cap = cv2.VideoCapture('./videos/cute_cat.mp4', cv2.CAP_ANY)

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        resized = cv2.resize(frame, (600, 1000))

        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        hsv  = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        lab  = cv2.cvtColor(resized, cv2.COLOR_BGR2Lab)

        cv2.imshow('BGR', resized)
        cv2.imshow('Gray', gray)
        cv2.imshow('HSV', hsv)
        cv2.imshow('Lab', lab)

        if cv2.waitKey(1) & 0xFF == 27:
            break

        time.sleep(0.01)
    else:
        break

cap.release()
cv2.destroyAllWindows()