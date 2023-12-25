from tkinter import Tk, Button, filedialog, Label, Canvas, Scale
from pathlib import Path
from PIL import ImageTk
from PIL import Image
import cv2
import numpy as np

def apply_gaussian_blur(image, kernel_size=5):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def find_intensity_gradient(blurred_image):
    gx = cv2.Sobel(np.float32(blurred_image), cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(np.float32(blurred_image), cv2.CV_64F, 0, 1, ksize=3)
    
    magnitude = np.sqrt(gx**2 + gy**2)
    orientation = np.arctan2(gy, gx) * (180 / np.pi)

    return magnitude, orientation

def non_maximum_suppression(magnitude, orientation):
    row, col = magnitude.shape
    output = np.zeros((row, col), dtype=np.float32)
    angle = orientation / 45 # adjust gradient angle

    for i in range(1, row - 1):
        for j in range(1, col - 1):
            # neighbors that are relevant for non-maximum suppression
            q = 255
            r = 255
            
            # interpolate pixels based on edge orientation
            # for diagonal edges:
            if (angle[i,j] < 22.5 and angle[i,j] >= 0) or \
               (angle[i,j] >= 157.5 and angle[i,j] < 202.5) or \
               (angle[i,j] >= 337.5 and angle[i,j] <= 360):
                q = magnitude[i, j+1]
                r = magnitude[i, j-1]
            # for vertical edges:
            elif (angle[i,j] >= 22.5 and angle[i,j] < 67.5) or \
                 (angle[i,j] >= 202.5 and angle[i,j] < 247.5):
                q = magnitude[i+1, j-1]
                r = magnitude[i-1, j+1]
            # for horizontal edges:
            elif (angle[i,j] >= 67.5 and angle[i,j] < 112.5) or\
                 (angle[i,j] >= 247.5 and angle[i,j] < 292.5):
                q = magnitude[i+1, j]
                r = magnitude[i-1, j]
            # for other diagonal edges:
            else:
                q = magnitude[i-1, j-1]
                r = magnitude[i+1, j+1]

            # non-maximum suppression
            if (magnitude[i,j] >= q) and (magnitude[i,j] >= r):
                output[i,j] = magnitude[i,j]
            else:
                output[i,j] = 0

    return output

def threshold(image, low_threshold_ratio=0.05, high_threshold_ratio=0.15):
    high_threshold = image.max() * high_threshold_ratio
    low_threshold = high_threshold * low_threshold_ratio
    
    m, n = image.shape
    res = np.zeros((m,n), dtype=np.uint8)
    
    strong_i, strong_j = np.where(image >= high_threshold)
    zeros_i, zeros_j = np.where(image < low_threshold)
    
    weak_i, weak_j = np.where((image <= high_threshold) & (image >= low_threshold))
    
    res[strong_i, strong_j] = 255
    res[weak_i, weak_j] = 75
    
    return (res, weak_i, weak_j)

def hysteresis(image, weak_i, weak_j):
    m, n = image.shape
    for i in range(1, m-1):
        for j in range(1, n-1):
            if image[i,j] == 75:
                if 255 in [image[i+1, j-1], image[i+1, j], image[i+1, j+1],
                            image[i, j-1], image[i, j+1],
                            image[i-1, j-1], image[i-1, j], image[i-1, j+1]]:
                    image[i, j] = 255
                else:
                    image[i, j] = 0
    return image

def canny_edge_detector(gray_image, low_threshold_ratio=0.05, high_threshold_ratio=0.15, gaussian_kernel_size=5):
    blurred_image = apply_gaussian_blur(gray_image, gaussian_kernel_size)
    gradient_magnitude, gradient_orientation = find_intensity_gradient(blurred_image)
    suppressed_image = non_maximum_suppression(gradient_magnitude, gradient_orientation)
    thresholded_image, weak_i, weak_j = threshold(suppressed_image, low_threshold_ratio, high_threshold_ratio)
    edges = hysteresis(thresholded_image, weak_i, weak_j)

    return edges