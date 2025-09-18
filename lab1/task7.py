import cv2

cap = cv2.VideoCapture(0)
ok, image = cap.read()
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
video_writer = cv2.VideoWriter('./videos/ouput.avi', fourcc, 30, (w, h))
while True:
    ok, image = cap.read()
    cv2.imshow('Webcam', image)
    video_writer.write(image)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()

