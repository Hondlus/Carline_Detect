import cv2
import numpy as np


cv2.namedWindow('edge_detection')
cv2.createTrackbar('minThreshold', 'edge_detection', 50, 1000, lambda x: x)
cv2.createTrackbar('maxThreshold', 'edge_detection', 100, 1000, lambda x: x)

img_color = cv2.imread('./testimg/img3.png')
height, width, _ = img_color.shape
img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
mask = np.zeros_like(img)

while 1:
    minThreshold = cv2.getTrackbarPos('minThreshold', 'edge_detection')
    maxThreshold = cv2.getTrackbarPos('maxThreshold', 'edge_detection')
    print('minThreshold, maxThreshold: ', minThreshold, maxThreshold)

    edge_img = cv2.Canny(img, minThreshold, maxThreshold)
    cv2.fillPoly(mask, np.array([[[int(0), int(0)], [0, int(height)], [width, int(height)], [int(width), int(0)]]]), 255)
    masked_edge_img = cv2.bitwise_and(edge_img, mask)
    # masked_img = cv2.bitwise_and(img, mask)
    cv2.imshow('masked_edge_img', masked_edge_img)
    # cv2.imshow('mask', masked_img)
    if cv2.waitKey(10) == ord('q'):
        break