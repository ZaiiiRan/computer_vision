import cv2

cap = cv2.VideoCapture('http://192.168.31.122:8080/video')
cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
while True:
    ok, image = cap.read()
    if ok:
        cv2.imshow('Video', image)

        if cv2.waitKey(1) & 0xFF == 27:
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()