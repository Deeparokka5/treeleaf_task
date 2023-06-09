import math
from typing import *

import cv2
import numpy as np

import matplotlib.pyplot as plt

def read_and_resize_image(img_path: str, scale_ratio: float = 0.2) -> List[any]:
    """This function reads an image from the given path, crops the image
    around the rectangles, and then resizes the image.

    Args:
        img_path (str): path of the image
        scale_ratio (float, optional): The ratio for the image to be scaled. Defaults to 0.2.

    Returns:
        List[any]: Resized image
    """
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  # original size (4094, 2898, 3)

    # cropping the image around the rectangles
    cropped_img = img[150:2150, 250:2700, :]  # cropped size (2000, 2450, 3)

    width = int(cropped_img.shape[1] * scale_ratio)
    height = int(cropped_img.shape[0] * scale_ratio)
    dim = (width, height)

    resized = cv2.resize(cropped_img, dim, interpolation=cv2.INTER_AREA)
    return resized


def get_corners(img, max_corners: int = 24) -> List:
    """This function finds all the corner coordinates of rectangles and lines from the image.

    Since in the image, we have 4 rectangles and 2 lines, i.e., 4*4 + 4*2 = 24 corners

    Args:
        img : Image array
        max_corners (int, optional): Max amount of corners to track. Defaults to 24.

    Returns:
        List : List of corners
    """
    if img.shape.__len__() > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.float32(img)
    corners = cv2.goodFeaturesToTrack(img, max_corners, 0.01, 10)
    corners = np.intp(corners)
    corners = [a.tolist() for a in corners.reshape(max_corners, 2)]
    return corners


def get_rectangle_bounding_box(img) -> List[List[int]]:
    """This function finds the bounding box for all the rectangles, i.e., 4, and
    returns the coordinates of the bounding box in (x, y, w, h) format.

    Args:
        img : Image array

    Returns:
        List[List[int]]: Coordinates of the bounding box
    """

    if img.shape.__len__() > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.bitwise_not(img)

    (contours, _) = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    rectangles = []
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if area > 50:
            rectangles.append((x, y, w, h))
    return rectangles


def get_len_of_lines_for_each_rect(rectangles, corners) -> Dict:
    """Find the length of lines for each rectangle

    Args:
        rectangles : Array of rectangle coordinates
        corners : Array of all the corner coordinates

    Returns:
        Dict: Dict of the length of lines with rectangle box coordinates
    """
    lengths = {}
    for rect in rectangles:
        (x, y, w, h) = rect
        c = [x + w // 2, y + h // 2]  # center of the rectangle

        # filter the points that are closest to the center of rectangles
        filtered = list(filter(lambda x: math.dist(c, x) < w, corners))

        # mapping points with their distances from the center
        ps = {int(math.dist(c, p)): p for p in filtered}
        ps = dict(sorted(ps.items()))

        # two closest points from the center are the coordinates of the line
        l1, l2 = list(ps.values())[:2]

        # length of the line
        d = int(math.dist(l1, l2))
        lengths[d] = rect

    # sorting as per the length of the line
    return dict(sorted(lengths.items()))


def generate_numbers_on_image(img, rectangles, corners) -> None:
    """Generating numbers on the image below the rectangles as per the length of lines
    from 1 to 4.

    And generates the final image with the name rectangle_numbering.jpg

    Args:
        img : Image array
        rectangles : Array of rectangle coordinates
        corners : Array of all the corner coordinates
    """

    # finding the length of lines for each rectangle
    lengths = get_len_of_lines_for_each_rect(rectangles=rectangles, corners=corners)

    # writing numbers on the image
    i = 1
    for k, v in lengths.items():

        font = cv2.FONT_HERSHEY_SIMPLEX  # font

        (x, y, w, h) = v
        org = (x + 45, y + h + 30)  # position of the text

        # fontScale
        fontScale = 1
        # Blue color in BGR
        color = (0, 0, 255)
        # Line thickness of 2 px
        thickness = 2

        # putting text on the image
        cv2.putText(img, f"{i}", org, font, fontScale, color, thickness, cv2.LINE_AA)
        i += 1

    print("Generating final numbered image....")
    cv2.imwrite("output_image_rectangle_numbering.jpg", img)
    plt.imshow(img, cmap='gray')
    plt.title("Numbering of rectangles")
    plt.show()


if __name__ == "__main__":
    image_path = "./input_image.jpg"

    # read, crop and resize image
    resized = read_and_resize_image(image_path, scale_ratio=0.25)

    # get bounding box of the rectangles, i.e., 4 rectangles
    rects = get_rectangle_bounding_box(resized)

    # get corner coordinates of all the shapes, i.e., 24 corners
    corners = get_corners(resized)

    # generate numbers and save the image
    generate_numbers_on_image(img=resized, rectangles=rects, corners=corners)
