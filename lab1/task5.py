import cv2

image = cv2.imread('./imgs/cat.jpg', cv2.IMREAD_COLOR)

cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
cv2.namedWindow('HSV', cv2.WINDOW_NORMAL)

cv2.imshow('Original', image)
cv2.imshow('HSV', cv2.cvtColor(image, cv2.COLOR_BGR2HSV))

cv2.waitKey(0)
cv2.destroyAllWindows()
