import matplotlib.pyplot as plt
from typing import *

import cv2
import numpy as np

# Path to the input image
image_path = "./input_image.jpg"

# Read and resize image
img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # original size (4094, 2898, 3)
cropped_img = img[150:2150, 250:2700, :]  # cropped size (2000, 2450, 3)
width = int(cropped_img.shape[1] * 0.25)
height = int(cropped_img.shape[0] * 0.25)
dim = (width, height)
resized = cv2.resize(cropped_img, dim, interpolation=cv2.INTER_AREA)

# Get contours
if resized.shape.__len__() > 2:
    resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
binary = cv2.bitwise_not(resized)
(contours, _) = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# Align rectangles
copy_img = resized.copy()
for i, c in enumerate(contours):
    # Calculate the area of each contour
    area = cv2.contourArea(c)
    # Skip the contour if the area is less than the 100 threshold
    if area < 100:
        continue

    (x, y, w, h) = cv2.boundingRect(c)

    # Find the minimum area and angle of alignment of the image
    rect = cv2.minAreaRect(c)  # center, width, height, and angle

    box = cv2.boxPoints(rect)
    box = np.intp(box)

    # Retrieve the key parameters of the rotated bounding box
    center = (int(rect[0][0]), int(rect[0][1]))
    width = int(rect[1][0])
    height = int(rect[1][1])
    angle = int(rect[2])

    # Mapping the angle
    if width < height:
        angle = -(90 - angle)
    else:
        angle = angle

    # As cv2.warpAffine expects shape in (length, height)
    shape = (resized.shape[1], resized.shape[0])

    matrix = cv2.getRotationMatrix2D(center=center, angle=angle, scale=1)
    image = cv2.warpAffine(src=resized, M=matrix, dsize=shape)

    copy_img[y:y+h, x:x+w] = image[y:y+h, x:x+w]

# Save the aligned image
print("Generating final numbered image....")
cv2.imwrite(f'output_image_rectangle_alignment.jpg', copy_img)
plt.imshow(copy_img, cmap='gray')
plt.title("Alignment of rectangles")
plt.show()