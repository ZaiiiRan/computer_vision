import cv2
import numpy as np

def build_kernel(size, sigma):
    matr = np.zeros((size, size), dtype=float)

    a = (size - size // 2) - 1
    b = a

    for x in range(size):
        for y in range(size):
            matr[x, y] = (1 / (2 * np.pi * sigma**2)) * np.exp(-(((x - a)**2 + (y - b)**2)/(2 * sigma**2)))
    
    return matr

def norm_kernel(kernel):
    return kernel / kernel.sum()

def gaussian_blur(kernel_size, blur_parameter, image):
    kernel = build_kernel(kernel_size, blur_parameter)
    kernel = norm_kernel(kernel)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    height, width = gray_image.shape
    padding = kernel_size // 2

    padded_image = cv2.copyMakeBorder(gray_image, padding, padding, padding, padding, cv2.BORDER_REFLECT)

    blurred_image = np.zeros_like(gray_image, dtype=float)
    for i in range(height):
        for j in range(width):
            region = padded_image[i:i+kernel_size, j:j+kernel_size]
            blurred_image[i, j] = np.sum(region * kernel)
    
    return np.clip(blurred_image, 0, 255).astype(np.uint8)

image = cv2.imread('../lab1/imgs/cat.png')

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.namedWindow('Original gray image', cv2.WINDOW_NORMAL)
cv2.imshow('Original gray image', gray_image)

kernel_sizes = [3, 7]
blur_parameters = [3, 15]

for size in kernel_sizes:
    for blur_parameter in blur_parameters:
        blurred_image = gaussian_blur(size, blur_parameter, image)
        
        windowName = f'Blurred gray image, kernel size: {size}, blur parameter: {blur_parameter}'
        cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
        cv2.imshow(windowName, blurred_image)

        blurred_image_by_cv2 = cv2.GaussianBlur(gray_image, (size, size), blur_parameter)
        windowNameByCv2 = f'Blurred by cv2 gray image, kernel size: {size}, blur parameter: {blur_parameter}'
        cv2.namedWindow(windowNameByCv2, cv2.WINDOW_NORMAL)
        cv2.imshow(windowNameByCv2, blurred_image_by_cv2)

cv2.waitKey(0)
cv2.destroyAllWindows()