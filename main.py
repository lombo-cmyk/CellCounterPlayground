# from PIL import Image
#
# img = Image.open('raw_data/10x_8.tif')
# img_size = (round(img.size[0]/4), round(img.size[1]/4))
#
# for _ in range(150, 160):
#
#     thresh = _
#     fn = lambda x: 255 if x > thresh else 0
#     r = img.convert('L').point(fn, mode='1')
#     r.save(f'foo_app_2_{_}.png')

import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import data
from skimage import filters
from skimage import exposure
from skimage import io
# read the image file
img = cv2.imread('raw_data/10x_8.tif')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
scale = 0.2
new_shape = (round(gray.shape[1]*scale), round(gray.shape[0]*scale))
gray = cv2.resize(gray, new_shape)
cv2.imwrite(f"tmp.png", gray)
skimage_image = io.imread(f"tmp.png")
io.imshow(skimage_image)
# fig, ax = filters.try_all_threshold(skimage_image, figsize=(10, 8), verbose=False)
# plt.show()
# gray.resize(new_shape)
kernel = np.ones((3, 3), np.uint8)
val = filters.threshold_mean(skimage_image)
val2 = filters.threshold_yen(skimage_image)
# plt.imshow(skimage_image > val, cmap='gray', interpolation='nearest')
# plt.subplot(211)
# plt.imshow(skimage_image > val, cmap='gray', interpolation='nearest')
# plt.axis('off')
# plt.subplot(212)
# plt.imshow(skimage_image > val2, cmap='gray', interpolation='nearest')
# plt.axis('off')
# plt.show()
new_img = skimage_image > 130
io.imsave("tmptmp.png", new_img)
new_img = cv2.imread("tmptmp.png", 0)
# ret, bw_img = cv2.threshold(gray, 152, 255, cv2.THRESH_BINARY)
# gray = cv2.GaussianBlur(new_img, (3, 3), 0)
# ret3, th3 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# plt.imshow(th3, 'gray')
# plt.show()
img = cv2.imread('raw_data/10x_8.tif')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
new_shape = (round(gray_img.shape[1]*scale), round(gray_img.shape[0]*scale))
gray_img = cv2.resize(gray_img, new_shape)
dupa, binary_img = cv2.threshold(gray_img, 130, 255, cv2.THRESH_BINARY)
binary_img = cv2.dilate(binary_img, kernel, iterations=2)
cv2.imshow("output", binary_img)
cv2.waitKey(0)
_, _, boxes, _ = cv2.connectedComponentsWithStats(~binary_img)
boxes = boxes[1:]
filtered_boxes = []
for x,y,w,h,pixels in boxes:
    if pixels < 1000000 and h < 100 and w < 100 and h > 5 and w > 5:
        filtered_boxes.append((x,y,w,h))

for x,y,w,h in filtered_boxes:
    cv2.rectangle(gray_img, (x,y), (x+w,y+h), (0,0,255),2)

plt.imshow(cv2.cvtColor(gray_img, cv2.COLOR_BGR2RGB))
plt.show()
maxRadius = int(15)
minRadius = int(1)

# if circles is not None:
#     # convert the (x, y) coordinates and radius of the circles to integers
#     circles = np.round(circles[0, :]).astype("int")
#     # loop over the (x, y) coordinates and radius of the circles
#     for (x, y, r) in circles:
#         # draw the circle in the output image, then draw a rectangle
#         # corresponding to the center of the circle
#         cv2.circle(gray, (x, y), r, (0, 255, 0), 4)
#         # cv2.rectangle(new_img, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
#     # show the output image
#     cv2.imshow("output", gray)
#     cv2.waitKey(0)
# for _ in range(10):
#     ret, bw_img = cv2.threshold(gray, 150+_, 255, cv2.THRESH_BINARY)
#     cv2.imwrite(f"black_white_base_{_}.tif", bw_img)
#     bw_img = cv2.dilate(bw_img, kernel, iterations=1)
#     bw_img = cv2.erode(bw_img, kernel, iterations=1)
#     bw_img = cv2.dilate(bw_img, kernel, iterations=1)
#     bw_img = cv2.erode(bw_img, kernel, iterations=3)
#
#     # converting to its binary form
#     # bw = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
#     cv2.imwrite(f"black_white_{_}.tif", bw_img)
# cv2.imshow("Binary", gray)
# cv2.waitKey(0)
# cv2.destroyAllWindows()