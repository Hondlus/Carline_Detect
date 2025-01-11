import cv2
import numpy as np


threshold = 0.2

# 计算斜率
def calculate_slope(line):
    x1, y1, x2, y2 = line[0]
    return (y2 - y1) / (x2 - x1)

# 过滤斜率相差较大的直线
def filter_lines(lines):

    slopes = [calculate_slope(line) for line in lines]
    while len(lines) > 0:
        mean = np.mean(slopes)
        diff = [abs(s - mean) for s in slopes]
        idx = np.argmax(diff)
        if diff[idx] > threshold:
            slopes.pop(idx)
            lines.pop(idx)
        else:
            break
    return lines

# 最小二乘直线拟合
def least_squares_fit(lines):
    x_coords = np.ravel([[line[0][0], line[0][2]] for line in lines])
    y_coords = np.ravel([[line[0][1], line[0][3]] for line in lines])
    poly = np.polyfit(x_coords, y_coords, 1)
    point_min = (np.min(x_coords), np.polyval(poly, np.min(x_coords)))
    point_max = (np.max(x_coords), np.polyval(poly, np.max(x_coords)))
    return np.array([point_min, point_max], dtype=np.int)


if __name__ == '__main__':

    img_color = cv2.imread('./testimg/img2.png')
    height, width, _ = img_color.shape
    img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(img)

    edge_img = cv2.Canny(img, 202, 696)
    # 提取roi感兴趣区域
    cv2.fillPoly(mask, np.array([[[int(width * 0.2), int(height * 0.1)], [0, int(height * 0.9)], [width, int(height * 0.9)], [int(width * 0.8), int(height * 0.1)]]]), 255)
    # cv2.fillPoly(mask, np.array([[[0, int(height * 0.5)], [0, int(height * 0.9)], [width, int(height * 0.9)], [width, int(height * 0.5)]]]), 255)
    masked_edge_img = cv2.bitwise_and(edge_img, mask)
    cv2.imshow('masked_edge_img', masked_edge_img)
    # 霍夫直线检测
    lines = cv2.HoughLinesP(masked_edge_img, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
    # 按照斜率划分车道线
    horizantal_lines = [line for line in lines if
                        0.1 >= calculate_slope(line) >= 0 or -0.1 <= calculate_slope(line) <= 0]
    # 可视化直线
    if horizantal_lines is not None:
        for line in horizantal_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img_color, (x1, y1), (x2, y2), (0, 255, 0), 2)


    # 最终拟合的直线
    result_line = least_squares_fit(horizantal_lines)
    cv2.line(img_color, result_line[0], result_line[1], (255, 0, 255), 5)

    cv2.imshow('result', img_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()