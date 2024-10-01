import cv2
import numpy as np


def find_grid(circles, rows, cols):
    circles = sorted(circles, key=lambda c: (c[1], c[0]))
    grid = []
    for i in range(0, len(circles), cols):
        row = sorted(circles[i:i + cols], key=lambda c: c[0])
        grid.append(row)
    return grid


def locate_filled_circles(grid, contours, min_overlap_ratio=0.5):
    answer_matrix = np.zeros((len(grid), len(grid[0])), dtype=int)
    for i, row in enumerate(grid):
        for j, (x, y, r) in enumerate(row):
            circle_area = np.pi * r * r
            for contour in contours:
                contour_area = cv2.contourArea(contour)
                mask = np.zeros((int(y + r * 2), int(x + r * 2)), dtype=np.uint8)
                cv2.circle(mask, (int(x), int(y)), int(r), 255, -1)
                overlap = cv2.bitwise_and(mask, cv2.drawContours(np.zeros_like(mask), [contour], 0, 255, -1))
                overlap_area = np.sum(overlap) / 255
                if overlap_area / circle_area > min_overlap_ratio:
                    answer_matrix[i, j] = 1
                    break
    return answer_matrix


def process_answer_sheet_p1(image_path):
    # Đọc ảnh
    img = cv2.imread(image_path)

    # Chuyển sang ảnh xám
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Áp dụng ngưỡng
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Áp dụng xử lý hình thái học
    kernel = np.ones((7, 7), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Tìm contours của các chấm đã tô
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Lọc contours dựa trên diện tích để loại bỏ nhiễu
    min_area = 100
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

    # Tìm các vòng tròn (ô trống)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                               param1=50, param2=30, minRadius=10, maxRadius=40)

    # Chuyển đổi kết quả sang dạng số nguyên
    circles = np.uint16(np.around(circles[0]))

    # Xác định số hàng và cột
    rows, cols = 4, 4

    # Tìm lưới từ các vòng tròn
    grid = find_grid(circles, rows, cols)


    # Xác định vị trí các điểm đã tô
    answer_matrix = locate_filled_circles(grid, filtered_contours)

    return answer_matrix

# Sử dụng hàm
# result = process_answer_sheet('/home/infcapital/Documents/yolov8/p3/cau6.png')
# print(result)