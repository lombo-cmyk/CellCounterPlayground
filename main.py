import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from datetime import datetime


def create_directories(upstream_folder_name: str):
    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, upstream_folder_name + '/')
    current_working_dir = os.path.join(results_dir, datetime.now().strftime(
        "%Y_%m_%d_%H_%M_%S/"))
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    os.makedirs(current_working_dir)
    return current_working_dir


def read_image(file: str):
    scale = 0.2
    img = cv2.imread(f'raw_data/{file}')
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return resize_image(gray_img, scale)


def resize_image(img: np.ndarray, scale: float):
    new_shape = round(img.shape[1] * scale), round(img.shape[0] * scale)
    return cv2.resize(img, new_shape)


def process_image_to_binary(img: np.ndarray, threshold: int,
                            kernel: np.ndarray):
    dupa, binary_img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    binary_img = ~binary_img
    binary_img = cv2.erode(binary_img, kernel, iterations=3)
    binary_img = cv2.dilate(binary_img, kernel, iterations=1)
    return binary_img


def find_cell_nucleus(img: np.ndarray):
    norm_one_percent_height = img.shape[0]/100
    norm_one_percent_width = img.shape[1]/100
    _, _, boxes, _ = cv2.connectedComponentsWithStats(img)
    boxes = boxes[1:]
    filtered_boxes = []
    for x, y, w, h, pixels in boxes:
        if pixels < 100000 and \
                h < 7 * norm_one_percent_height and \
                w < 3 * norm_one_percent_width and \
                h > norm_one_percent_height and \
                w > norm_one_percent_width:
            if 0.5 < h / w < 1.5:
                filtered_boxes.append((x, y, w, h))
    return filtered_boxes


def main():
    working_dir = create_directories("tmp")
    kernel = np.ones((2, 2), np.uint8)
    binarization_threshold = 140
    raw_image_name = "10x_8.tif"
    image_name = raw_image_name.split(".")[0]

    gray_img = read_image(raw_image_name)
    binary_img = process_image_to_binary(gray_img, binarization_threshold,
                                         kernel)
    cv2.imwrite(f"{working_dir}binary_{image_name}.png", binary_img)
    boxes = find_cell_nucleus(binary_img)
    for x, y, w, h in boxes:
        cv2.rectangle(gray_img, (x, y), (x+w, y+h), (0, 0, 255), 2)

    plt.imsave(f"{working_dir}nucleus_{image_name}.png",
               cv2.cvtColor(gray_img, cv2.COLOR_BGR2RGB))
    print(f"for: {len(boxes)}")


if __name__ == "__main__":
    main()
