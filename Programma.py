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

class ContourImageProcessor:
    def __init__(self, master):
        self.master = master
        self.master.geometry("800x600")
        self.master.title("Contour Image Creator")
        self.master.configure(bg="white")

        self.pil_image = None
        self.image_path = None
        self.contour_image = None
        self.tk_image = None  # Store tk_image as an instance variable

        self.create_widgets()

    def create_widgets(self):
        self.canvas = Canvas(self.master, width=600, height=400)
        self.canvas.grid(row=0, column=0, padx=20, pady=20, rowspan=5)

        select_button = Button(self.master, text="Select Image", command=self.select_image)
        select_button.grid(row=0, column=1, padx=20, pady=10)

        create_button = Button(self.master, text="Create Contour Image", command=self.create_contour_image)
        create_button.grid(row=1, column=1, padx=20, pady=10)

        self.threshold_label = Label(self.master, text="Canny Threshold:")
        self.threshold_label.grid(row=2, column=1, padx=20, pady=10)

        self.threshold_scale = Scale(self.master, from_=0, to=50, orient="horizontal", length=200)
        self.threshold_scale.set(100)
        self.threshold_scale.grid(row=3, column=1, padx=20, pady=10)

        show_original_button = Button(self.master, text="Show Original Image", command=self.show_original_image)
        show_original_button.grid(row=4, column=1, padx=20, pady=10)

        save_button = Button(self.master, text="Save Image", command=self.save_image)
        save_button.grid(row=5, column=1, padx=20, pady=10)

    def select_image(self):
        file_types = [("JPEG files", "*.jpg;*.jpeg"), ("PNG files", "*.png")]
        input_image_path = Path(filedialog.askopenfilename(filetypes=file_types))

        if not input_image_path.exists():
            print('The specified input file does not exist')
            return

        self.image_path = input_image_path
        self.pil_image = Image.open(input_image_path)
        self.display_image(self.pil_image)

    def apply_canny_algorithm(self, image, canny_threshold):
        # Преобразовать изображение PIL в массив NumPy
        np_image = np.array(image)
        # Обработка изображения
        gray_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2GRAY)
        edges = canny_edge_detector(gray_image, canny_threshold / 255, (canny_threshold * 2) / 255)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_image = np.zeros_like(np_image)
        cv2.drawContours(contour_image, contours, -1, (255, 255, 255), 1)
        return contour_image

    def create_contour_image(self):
        if not self.pil_image:
            print("Please select an image first.")
            return
        
        canny_threshold = self.threshold_scale.get()
        self.contour_image = self.apply_canny_algorithm(self.pil_image, canny_threshold)
        self.display_image(Image.fromarray(self.contour_image))

    def show_original_image(self):
        if self.pil_image:
            self.display_image(self.pil_image)

    def save_image(self):
        if self.contour_image is not None:
            file_types = [("JPEG files", "*.jpg;*.jpeg"), ("PNG files", "*.png")]
            output_file_path = Path(filedialog.asksaveasfilename(filetypes=file_types, defaultextension=".png"))

            if output_file_path:
                Image.fromarray(self.contour_image).save(output_file_path)
                print(f"Image saved successfully to {output_file_path}")
        else:
            print("No contour image to save. Create one first.")

    def display_image(self, image):
        # Define the region of interest (ROI) coordinates (left, top, right, bottom)
        roi_coords = (100, 50, 500, 350)

        # Resize the image based on the ROI
        image = image.resize((roi_coords[2] - roi_coords[0], roi_coords[3] - roi_coords[1]))

        self.tk_image = ImageTk.PhotoImage(image)
        self.canvas.config(width=self.tk_image.width(), height=self.tk_image.height())
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)
        self.canvas.image = self.tk_image