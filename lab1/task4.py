import cv2

cap = cv2.VideoCapture('./videos/cute_cat.mp4')

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('./videos/new_cat.avi', fourcc, 30, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        out.write(frame)
    else:
        break

cap.release()
out.release()
