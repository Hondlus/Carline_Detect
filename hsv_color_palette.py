import cv2
import numpy as np

def nothing(x):
    pass

img = cv2.imread('./testimg/img3.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
window_name = 'HSV COLOR Palette'
cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

cv2.createTrackbar('H', window_name, 0, 179, nothing)
cv2.createTrackbar('H2', window_name, 179, 179, nothing)
cv2.createTrackbar('S', window_name, 0, 255, nothing)
cv2.createTrackbar('S2', window_name, 18, 255, nothing)
cv2.createTrackbar('V', window_name, 155, 255, nothing)
cv2.createTrackbar('V2', window_name, 255, 255, nothing)

while 1:

    h = cv2.getTrackbarPos('H', window_name)
    h2 = cv2.getTrackbarPos('H2', window_name)
    s = cv2.getTrackbarPos('S', window_name)
    s2 = cv2.getTrackbarPos('S2', window_name)
    v = cv2.getTrackbarPos('V', window_name)
    v2 = cv2.getTrackbarPos('V2', window_name)

    lower_white = np.array([h, s, v])
    upper_white = np.array([h2, s2, v2])
    mask = cv2.inRange(img, lower_white, upper_white)

    cv2.imshow(window_name, mask)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()