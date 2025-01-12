import cv2
import numpy as np


def track_white(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # lower_white = np.array([0, 0, 212])
    # upper_white = np.array([131, 255, 255])
    lower_white = np.array([0, 0, 155])
    upper_white = np.array([179, 18, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    return mask

def cal_angle(x1, y1, x2, y2):
    if x2 - x1 == 0:
        angle = 90
        # print("angle90: ", angle)
    elif y2 - y1 == 0:
        angle = 0
        # print("angle0: ", angle)
    else:
        k = -(y2 - y1) / (x2 - x1)
        # print("k", k)
        angle = int(np.arctan(k) * 180 / np.pi)
        # print("anglek: ", angle)
    return angle

def least_squares_fit(lines):
    x_coords = np.ravel([[line[0][0], line[0][2]] for line in lines])
    y_coords = np.ravel([[line[0][1], line[0][3]] for line in lines])
    poly = np.polyfit(x_coords, y_coords, 1)
    point_min = (np.min(x_coords), np.polyval(poly, np.min(x_coords)))
    point_max = (np.max(x_coords), np.polyval(poly, np.max(x_coords)))
    return np.array([point_min, point_max], dtype=np.int)

if __name__ == '__main__':
    line_list = []
    image = cv2.imread('./testimg/img.png')
    mask = track_white(image)
    cv2.imshow('mask', mask)

    median_filtered_image = cv2.medianBlur(mask, 3)
    cv2.imshow('median', median_filtered_image)

    lines = cv2.HoughLinesP(median_filtered_image, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

    # 可视化直线
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # 筛选想要的直线
            angle = cal_angle(x1, y1, x2, y2)
            if angle == 0:
            # if -1 <= angle <= 1:
                line_list.append(line)
                cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            else:
                cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    result_line = least_squares_fit(line_list)
    cv2.line(image, result_line[0], result_line[1], (255, 0, 255), 2)
    cv2.imshow('image', image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()