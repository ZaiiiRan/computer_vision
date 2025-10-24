import cv2
import numpy as np

blur_parameter = 3

def build_kernel(size, sigma):
    matr = np.zeros((size, size), dtype=float)

    a = (size - size // 2) - 1
    b = a

    for x in range(size):
        for y in range(size):
            matr[x, y] = (1 / (2 * np.pi * sigma**2)) * np.exp(-(((x - a)**2 + (y - b)**2)/(2 * sigma**2)))
    
    return matr

kernel3 = build_kernel(3, blur_parameter)
kernel5 = build_kernel(5, blur_parameter)
kernel7 = build_kernel(7, blur_parameter)

print('Ядро свертки 3x3: ', kernel3)
print('Ядро свертки 5x5: ', kernel5)
print('Ядро свертки 7x7: ', kernel7)