import os

import cv2
import numpy as np


def extract_roi(img_color):
    height, width, _ = img_color.shape
    img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(img)
    # 提取roi感兴趣区域
    cv2.fillPoly(mask, np.array([[[int(width * 0.2), int(height * 0.2)], [0, int(height * 1.0)], [width, int(height * 1.0)], [int(width * 0.8), int(height * 0.2)]]]), 255)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img

def canny_detect(img_color):
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    canny_image = cv2.Canny(img_gray, 100, 200)
    resize_canny = cv2.resize(canny_image, (480, 480))
    cv2.imshow('resize_canny', resize_canny)
    return canny_image

def track_white(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # lower_white = np.array([0, 0, 212])
    # upper_white = np.array([131, 255, 255])
    lower_white = np.array([0, 0, 155])
    upper_white = np.array([179, 18, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)

    resize_mask = cv2.resize(mask, (480, 480))
    # cv2.imshow('mask', mask)
    cv2.imshow('resize_mask', resize_mask)

    median_filtered_image = cv2.medianBlur(mask, 3)
    resize_median = cv2.resize(median_filtered_image, (480, 480))
    # cv2.imshow('median', median_filtered_image)
    cv2.imshow('resize_median', resize_median)

    return median_filtered_image

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
#多条直线拟合成一条
def least_squares_fit(lines):
    x_coords = np.ravel([[line[0][0], line[0][2]] for line in lines])
    y_coords = np.ravel([[line[0][1], line[0][3]] for line in lines])
    poly = np.polyfit(x_coords, y_coords, 1)
    point_min = (np.min(x_coords), np.polyval(poly, np.min(x_coords)))
    point_max = (np.max(x_coords), np.polyval(poly, np.max(x_coords)))
    return np.array([point_min, point_max], dtype=np.int)

if __name__ == '__main__':
    line_list = []
    img_path = "C:/Users/dxw-user/Desktop/pachong/tingzhixian/"

    for img_name in os.listdir(img_path):
        print(img_path + img_name)
        image = cv2.imread(img_path + img_name)
        height, width = image.shape[:2]
        # print("height, width: ", height, width)

        # 检测白色并提取二值化图片
        erzhi_image = track_white(image)
        # canny边缘检测
        # erzhi_image = canny_detect(image)

        lines = cv2.HoughLinesP(erzhi_image, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

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
                    # pass
                else:
                    cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    # pass

        if len(line_list) != 0:
            result_line = least_squares_fit(line_list)
            cv2.line(image, result_line[0], result_line[1], (0, 255, 0), 2)

        resize_result = cv2.resize(image, (480, 480))
        cv2.imshow('resize_result', resize_result)
        # cv2.imwrite('C:/Users/dxw-user/Desktop/carline3.jpg', image)

        line_list.clear()

        if cv2.waitKey(0) & 0xFF == 27:
            break

    cv2.destroyAllWindows()