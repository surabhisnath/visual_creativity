import cv2
import numpy as np
import os

smoothness_sum = 0
i = 0
path = "../data/adults/stimuli_R/"
for im in os.listdir(path):
    if im != ".DS_Store":
        image = cv2.imread(path + im, cv2.IMREAD_GRAYSCALE)
        edges = cv2.Canny(image, 100, 200)
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Measure contour smoothness (example: using arc length to contour length ratio)
        smoothness_values = []
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            area = cv2.contourArea(contour)
            if area > 0:
                smoothness = perimeter**2 / (4 * np.pi * area)
                smoothness_values.append(smoothness)
        final_smoothness = np.mean(smoothness_values) if smoothness_values else None
        smoothness_sum += final_smoothness

        i += 1

print("Mean roughness", smoothness_sum / i)
