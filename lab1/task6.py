import cv2
import os

cap = cv2.VideoCapture(0)
ret, image = cap.read()
if not ret:
    os.exit(1)
cap.release()

color = (0, 0, 255)
thick = 2

height, width = image.shape[:2]

# вертикальный прямоугольник
rect_1_width = 30
rect_1_height = 200
rect_1_corner_1 = (width // 2 - rect_1_width // 2, height // 2 - rect_1_height // 2)
rect_1_corner_2 = (width // 2 + rect_1_width // 2, height // 2 + rect_1_height // 2)

# горизонтальный прямоугольник
rect_2_width = 170
rect_2_height = 35
rect_2_corner_1 = (width // 2 - rect_2_width // 2, height // 2 - rect_2_height // 2)
rect_2_corner_2 = (width // 2 + rect_2_width // 2, height // 2 + rect_2_height // 2)

# содержимое горизонтального прямоугольника
rect_2_content = image[rect_2_corner_1[1]:rect_2_corner_2[1], rect_2_corner_1[0]:rect_2_corner_2[0]]
blur = cv2.GaussianBlur(rect_2_content, (15, 15), 10)

cv2.rectangle(image, rect_1_corner_1, rect_1_corner_2, color, thick)
image[rect_2_corner_1[1]:rect_2_corner_2[1], rect_2_corner_1[0]:rect_2_corner_2[0]] = blur
cv2.rectangle(image, rect_2_corner_1, rect_2_corner_2, color, thick)

cv2.imshow('Result', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
