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

    blurred_image = cv2.GaussianBlur(gray_image, (7, 7), 4)

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

    print(f'Матрица значений длин:\n{grad_magnitude}\n')
    print(f'Матрица значений углов:\n{angle}\n')

    non_maximum_suppression = np.zeros_like(grad_magnitude, dtype=np.uint8)
    for i in range(1, grad_magnitude.shape[0] - 1):
        for j in range(1, grad_magnitude.shape[1] - 1):
            direction = angle[i, j]
            magnitude = grad_magnitude[i, j]

            if direction in [0, 4]:
                neighbors = [grad_magnitude[i, j - 1], grad_magnitude[i, j + 1]]
            elif direction in [1, 5]:
                neighbors = [grad_magnitude[i - 1, j + 1], grad_magnitude[i + 1, j - 1]]
            elif direction in [2, 6]:
                neighbors = [grad_magnitude[i - 1, j], grad_magnitude[i + 1, j]]
            else :
                neighbors = [grad_magnitude[i - 1, j - 1], grad_magnitude[i + 1, j + 1]]

            if magnitude > max(neighbors):
                non_maximum_suppression[i, j] = 255
            else:
                non_maximum_suppression[i, j] = 0
    
    cv2.namedWindow("Non maximum suppression", cv2.WINDOW_NORMAL)
    cv2.imshow("Non maximum suppression", non_maximum_suppression)
    
    max_grad = np.max(grad_magnitude)
    low_level = max_grad // 13
    high_level = max_grad // 17

    strong_edges = (grad_magnitude >= high_level)
    weak_edges = ((grad_magnitude >= low_level) & (grad_magnitude < high_level))

    result = np.zeros_like(non_maximum_suppression, dtype=np.uint8)
    result[strong_edges & (non_maximum_suppression == 255)] = 255

    for i in range(1, grad_magnitude.shape[0] - 1):
        for j in range(1, grad_magnitude.shape[1] - 1):
            if weak_edges[i, j] and non_maximum_suppression[i, j] == 255:
                region = result[i-1:i+2, j-1:j+2]
                if np.any(region == 255):
                    result[i, j] = 255
    
    cv2.namedWindow('Double threshold filtering', cv2.WINDOW_NORMAL)
    cv2.imshow("Double threshold filtering", result)
    
    contoured_image = image.copy()
    contoured_image[result == 255] = [0, 255, 0]

    cv2.namedWindow('Contoured', cv2.WINDOW_NORMAL)
    cv2.imshow("Contoured", contoured_image)


highlight_borders('./cat.jpg')
cv2.waitKey(0)
cv2.destroyAllWindows()
