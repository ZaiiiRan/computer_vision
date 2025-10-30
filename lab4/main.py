import cv2
import numpy

def highlight_borders(image_path):
    image = cv2.imread(image_path)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 3)

    cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
    cv2.imshow("Original", image)
    cv2.namedWindow('Blurred', cv2.WINDOW_NORMAL)
    cv2.imshow("Blurred", blurred_image)

highlight_borders('./cat.png')
cv2.waitKey(0)
cv2.destroyAllWindows()
