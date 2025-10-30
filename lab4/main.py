import cv2
import numpy as np

def angle_round(grad_x, grad_y, tg):
    angle = np.zeros_like(grad_x, dtype=np.uint8)

    angle[((grad_x > 0) & (grad_y < 0) & (tg < -2.414)) | ((grad_x < 0) & (grad_y < 0) & (tg > 2.414))] = 0
    angle[(grad_x > 0) & (grad_y < 0) & (tg >= -2.414) & (tg < -0.414)] = 1
    angle[((grad_x > 0) & (grad_y < 0) & (tg >= -0.414)) | ((grad_x > 0) & (grad_y > 0) & (tg < 0.414))] = 2
    angle[(grad_x > 0) & (grad_y > 0) & (tg >= 0.414) & (tg < 2.414)] = 3
    angle[((grad_x > 0) & (grad_y > 0) & (tg >= 2.414)) | ((grad_x < 0) & (grad_y > 0) & (tg <= -2.414))] = 4
    angle[(grad_x < 0) & (grad_y > 0) & (tg > -2.414) & (tg <= -0.414)] = 5
    angle[((grad_x < 0) & (grad_y > 0) & (tg > -0.414)) | ((grad_x < 0) & (grad_y < 0) & (tg < 0.414))] = 6
    angle[(grad_x < 0) & (grad_y < 0) & (tg >= 0.414) & (tg < 2.414)] = 7

    return angle

def highlight_borders(image_path):
    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 3)

    cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
    cv2.imshow("Original", image)
    cv2.namedWindow('Blurred', cv2.WINDOW_NORMAL)
    cv2.imshow("Blurred", blurred_image)

    sobel_kernel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float64)

    sobel_kernel_y = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ], dtype=np.float64)

    grad_x = np.zeros_like(blurred_image, dtype=np.float64)
    grad_y = np.zeros_like(blurred_image, dtype=np.float64)

    padded_image = np.pad(blurred_image, ((1, 1), (1, 1)), mode='reflect')

    for i in range(height):
        for j in range(width):
            region = padded_image[i:i+3, j:j+3]
            grad_x[i, j] = np.sum(region * sobel_kernel_x)
            grad_y[i, j] = np.sum(region * sobel_kernel_y)
    
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    tg = grad_y / np.where(grad_x == 0, 1e-6, grad_x)

    angle = angle_round(grad_x, grad_y, tg)

    print(f'Матрица длин градиентов:\n{grad_magnitude}\n')
    print(f'Матрица углов градиентов:\n{angle}\n')



highlight_borders('./cat.jpg')
cv2.waitKey(0)
cv2.destroyAllWindows()
