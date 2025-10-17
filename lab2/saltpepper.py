import cv2
import numpy as np

def salt_pepper(image): 
    h, w = image.shape[:2]
    h = int(h)
    w = int(w)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    for i in range(h):
        for j in range(w):
            random = np.random.random()
            if random < 0.03:  
                salt_or_pepper = np.random.randint(0, 2)
                if salt_or_pepper == 0:
                    gray_image[i, j] = 0   
                else:
                    gray_image[i, j] = 255
    return gray_image

# cap = cv2.VideoCapture(0)

# ret, frame = cap.read()

frame = cv2.imread('../lab1/imgs/cat.png', cv2.IMREAD_COLOR)

cv2.imshow('original', frame)
cv2.imshow('salt-pepper', salt_pepper(frame))

cv2.waitKey(0)
cv2.destroyAllWindows()
