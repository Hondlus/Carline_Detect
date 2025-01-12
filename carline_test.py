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

# 计算直线的角度和长度
def cal_angle_distance(x1, y1, x2, y2):
    x_bias = x2 - x1
    y_bias = y2 - y1

    distance = np.sqrt(x_bias ** 2 + y_bias ** 2)

    if x_bias == 0:
        angle = 90
        # print("angle90: ", angle)
    elif y_bias == 0:
        angle = 0
        # print("angle0: ", angle)
    else:
        k = -y_bias / x_bias
        # print("k", k)
        angle = int(np.arctan(k) * 180 / np.pi)
        # print("anglek: ", angle)
    return angle, distance

def least_squares_fit(lines):
    x_coords = np.ravel([[line[0][0], line[0][2]] for line in lines])
    y_coords = np.ravel([[line[0][1], line[0][3]] for line in lines])
    poly = np.polyfit(x_coords, y_coords, 1)
    point_min = (np.min(x_coords), np.polyval(poly, np.min(x_coords)))
    point_max = (np.max(x_coords), np.polyval(poly, np.max(x_coords)))
    return np.array([point_min, point_max], dtype=np.int)

if __name__ == '__main__':
    line_list = []
    image = cv2.imread('./testimg/123.jpg')
    height, width = image.shape[:2]
    # print("height, width: ", height, width)
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
            angle, distance = cal_angle_distance(x1, y1, x2, y2)
            if angle == 0 and width * 0.25 <= distance <= width:
            # if -1 <= angle <= 1:
                line_list.append(line)
                # print(distance)
                cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            else:
                cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                # pass

    if len(line_list) != 0:
        result_line = least_squares_fit(line_list)
        cv2.line(image, result_line[0], result_line[1], (0, 255, 0), 2)
    cv2.imshow('image', image)

    line_list.clear()

    cv2.waitKey(0)
    cv2.destroyAllWindows()