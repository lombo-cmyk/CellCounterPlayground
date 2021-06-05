import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from datetime import datetime
from typing import Tuple


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


def cut_cell_nucleus(img: np.ndarray, boxes: list):
    circles = [(int(x+w/2), int(y+h/2), int((h+w)/3)) for x, y, w, h in boxes]
    for x, y, r in circles:
        tmp_img = cv2.circle(img, (x, y), r, 0, -1)
        tmp_img, corners = prepare_tmp_image(x, y, r, tmp_img)
        dominant_color = find_dominant_color(tmp_img)
        cv2.circle(img, (x, y), r, dominant_color, -1)
        img = blur_circle(img, corners, r)
    return img


def prepare_tmp_image(x: int, y: int, r: int, img: np.ndarray):
    one_height = int(img.shape[0]/100)
    one_width = int(img.shape[1]/100)
    corner_x = x - r
    corner_y = y - r

    x_begin = tmp if (tmp := corner_x - 7 * one_width) > 0 else (tmp := 0)
    x_end = tmp if (tmp := corner_x + 7 * one_width) <= img.shape[1] else(
        tmp := img.shape[1])
    y_begin = tmp if (tmp := corner_y - 7 * one_height) > 0 else (tmp := 0)
    y_end = tmp if (tmp := corner_y + 7 * one_height) <= img.shape[0] else(
        tmp := img.shape[0])

    new_image = img[y_begin:y_end, x_begin:x_end]
    return new_image, (corner_x, corner_y)


def find_dominant_color(img: np.ndarray):
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    non_black_index = 20
    if hist.argmax() > non_black_index:
        dominant_val = hist.argmax()
    else:
        dominant_val = 0
        counts = 0
        for i in range(non_black_index, len(hist)):
            if hist[i][0] > counts:
                dominant_val = i
                counts = int(hist[i][0])
    return int(dominant_val)


def blur_circle(img: np.ndarray, corners: Tuple[int, int], r: int):
    x = corners[0]
    y = corners[1]
    length = 2 * r
    image_to_blur = img[y:y+length, x:x+length]
    image_to_blur = cv2.bilateralFilter(image_to_blur, 9, 75, 75)
    img[y:y+length, x:x+length] = image_to_blur
    return img


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

    tmp_image = gray_img.copy()
    for x, y, w, h in boxes:
        cv2.rectangle(tmp_image, (x, y), (x+w, y+h), (0, 0, 255), 2)
    plt.imsave(f"{working_dir}nucleus_{image_name}.png",
               cv2.cvtColor(tmp_image, cv2.COLOR_BGR2RGB))

    grey_cut_nucleus = cut_cell_nucleus(gray_img, boxes)
    cv2.imwrite(f"{working_dir}cut_nucleus_{image_name}.png", grey_cut_nucleus)
    gaus = cv2.GaussianBlur(grey_cut_nucleus, (3, 3), 0)
    edges_0 = cv2.Canny(gaus, 20, 80)
    plt.imshow(~edges_0, cmap='gray')
    plt.show()

if __name__ == "__main__":
    main()
