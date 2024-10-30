import cv2
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

# Load the drawing image

ink_densities = []
i = 0
path = "../data/adults/stimuli_R/"
for im in os.listdir(path):
    if im != ".DS_Store":
        # image = cv2.imread(path + im, cv2.IMREAD_GRAYSCALE)

        # # Ink on Paper Analysis
        # _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
        # ink_density = np.sum(binary == 255) / binary.size

        img = Image.open(path + im).convert("L")
        # img.show()
        pixels = img.getdata()
        arr = []
        for i in pixels:
            arr.append(i)

        ink_ratio = len([1 for i in pixels if i < 250]) / len(pixels)
        ink_densities.append(ink_ratio)
        i += 1

plt.hist(ink_densities)
plt.show()
