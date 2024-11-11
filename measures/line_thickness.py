import cv2
import numpy as np
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt
from PIL import Image
import matplotlib.pyplot as plt
import os

line_thicknesses = []
i = 0
path = "../data/children/stimuli_R/"
for im in os.listdir(path):
    if im != ".DS_Store":
        image = cv2.imread(path + im, cv2.IMREAD_GRAYSCALE)

        _, binary = cv2.threshold(
            image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # Skeletonize the binary image
        skeleton = skeletonize(binary // 255).astype(np.uint8)

        # Compute the distance transform on the binary image
        distance_transform = distance_transform_edt(binary)

        # Get the thickness values by mapping skeleton pixels to the distance transform
        thickness_values = (
            distance_transform[skeleton > 0] * 2
        )  # Multiply by 2 for full thickness

        # Calculate the average line thickness
        average_thickness = thickness_values.mean()
        line_thicknesses.append(average_thickness)

print(np.mean(line_thicknesses))
