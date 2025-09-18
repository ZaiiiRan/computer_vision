import cv2
import os
import math

cap = cv2.VideoCapture(0)
ret, image = cap.read()
if not ret:
    os.exit(1)
cap.release()

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

def get_nearest_color(pixel):
    b, g, r = pixel.astype(int)

    red_dist = math.sqrt((r - 255) ** 2 + (g - 0) ** 2 + (b - 0) ** 2)
    green_dist = math.sqrt((r - 0) ** 2 + (g - 255) ** 2 + (b - 0) ** 2)
    blue_dist = math.sqrt((r - 0) ** 2 + (g - 0) ** 2 + (b - 255) ** 2)
    min_dist = min(red_dist, green_dist, blue_dist)

    if min_dist == red_dist:
        return (0, 0, 255)
    elif min_dist == green_dist:
        return (0, 255, 0)
    else:
        return (255, 0, 0)

color = get_nearest_color(image[height // 2, width // 2])

cv2.rectangle(image, rect_1_corner_1, rect_1_corner_2, color, -1)
cv2.rectangle(image, rect_2_corner_1, rect_2_corner_2, color, -1)

cv2.imshow('Result', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
