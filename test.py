import cv2 
from matplotlib import pyplot as plt
import numpy as np

IMG_ROOT = 'image/'

source_img_1 = cv2.imread(IMG_ROOT + 'vita1.jpg')
source_img_2 = cv2.imread(IMG_ROOT + 'vita2.jpg')
images = [source_img_1,source_img_2]

print(images)

stitcher = cv2.Stitcher_create()
(status, stitched) = stitcher.stitch(images)

cv2.imwrite('stitched.jpg', stitched)
cv2.imshow("Stitched", stitched)
cv2.waitKey(0)
