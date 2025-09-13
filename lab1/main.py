import cv2

image_jpg = cv2.imread("./imgs/cat.jpg", cv2.IMREAD_COLOR)
image_png = cv2.imread("./imgs/cat.png", cv2.IMREAD_GRAYSCALE)
image_gif = cv2.imread("./imgs/cat.gif", cv2.IMREAD_REDUCED_COLOR_2)

cv2.namedWindow('Cat JPG', cv2.WINDOW_NORMAL)
cv2.imshow('Cat JPG', image_jpg)

cv2.namedWindow('Cat PNG', cv2.WINDOW_FULLSCREEN)
cv2.imshow('Cat PNG', image_png)

cv2.namedWindow('Cat GIF', cv2.WINDOW_AUTOSIZE)
cv2.imshow('Cat GIF', image_gif)

cv2.waitKey(0)
cv2.destroyAllWindows()
