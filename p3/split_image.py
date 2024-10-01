import cv2
import numpy as np
import matplotlib.pyplot as plt


def show_image(img, title):
    plt.figure(figsize=(10, 5))
    if len(img.shape) == 2:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()


def split_image(image_path):
    # Đọc hình ảnh
    img = cv2.imread(image_path)
    show_image(img, "1. Ảnh gốc")

    # Chuyển đổi sang ảnh grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    show_image(gray, "2. Ảnh grayscale")

    # Áp dụng ngưỡng để tạo ảnh nhị phân
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    show_image(binary, "3. Ảnh nhị phân")

    # Tìm các đường kẻ dọc
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, binary.shape[0] // 2))
    detect_vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    show_image(detect_vertical, "4. Phát hiện đường kẻ dọc")

    # Tìm các đường thẳng
    lines = cv2.HoughLinesP(detect_vertical, 1, np.pi / 180, threshold=100, minLineLength=binary.shape[0] // 2,
                            maxLineGap=20)

    # Vẽ các đường thẳng lên ảnh gốc
    img_with_lines = img.copy()
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img_with_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)
    show_image(img_with_lines, "5. Ảnh với các đường thẳng được phát hiện")

    # Sắp xếp các đường thẳng theo tọa độ x
    sorted_lines = sorted(lines, key=lambda line: line[0][0])
    split_points = [0] + [line[0][0] for line in sorted_lines] + [img.shape[1]]

    parts = []
    for i in range(len(split_points) - 1):
        start_x = split_points[i]
        end_x = split_points[i + 1]
        if end_x - start_x > 20:
            part = img[:, start_x:end_x]
            parts.append(part)
            show_image(part, f"6. Phần ảnh {i + 1}")

    return parts


def detect_circles_and_create_grid(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Cắt bớt phần trên, trái và dưới của ảnh để loại bỏ text
    height, width = gray.shape
    gray = gray[int(height * 0.17):int(height * 0.97), int(width * 0.1):]
    img = img[int(height * 0.17):int(height * 0.97), int(width * 0.1):]
    # show_image(gray, "1. Ảnh grayscale")

    # Áp dụng Gaussian blur để giảm nhiễu
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    # show_image(blurred, "2. Ảnh sau khi làm mờ")

    # Phát hiện hình tròn
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                               param1=50, param2=30, minRadius=8, maxRadius=20)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        img_with_circles = img.copy()

        all_circles = []
        filled_circles = []

        for circle in circles[0, :]:
            x, y, r = circle
            all_circles.append((x, y))
            cv2.circle(img_with_circles, (x, y), r, (0, 255, 0), 2)

            # Kiểm tra màu sắc bên trong hình tròn
            mask = np.zeros(gray.shape, np.uint8)
            cv2.circle(mask, (x, y), r - 2, 255, -1)
            mean_color = cv2.mean(img, mask=mask)[:3]  # Lấy giá trị trung bình của kênh BGR

            # Kiểm tra xem hình tròn có được tô màu đen không
            if np.mean(mean_color) < 100:  # Ngưỡng này có thể điều chỉnh
                filled_circles.append((x, y))
                cv2.circle(img_with_circles, (x, y), r, (0, 0, 255), -1)

        # show_image(img_with_circles, "3. Ảnh với các hình tròn được phát hiện")

        # Sắp xếp các hình tròn theo tọa độ y
        all_circles.sort(key=lambda c: c[1])

        # Xác định các hàng
        rows = []
        current_row = [all_circles[0]]
        for circle in all_circles[1:]:
            if abs(circle[1] - current_row[-1][1]) > r:  # Sử dụng bán kính làm ngưỡng
                rows.append(current_row)
                current_row = [circle]
            else:
                current_row.append(circle)
        rows.append(current_row)

        # Sắp xếp các hình tròn trong mỗi hàng theo tọa độ x
        for row in rows:
            row.sort(key=lambda c: c[0])

        # Xác định số cột tối đa
        max_cols = max(len(row) for row in rows)

        # Tạo lưới
        grid = np.zeros((len(rows), max_cols), dtype=int)

        for i, row in enumerate(rows):
            for j, (x, y) in enumerate(row):
                if (x, y) in filled_circles:
                    grid[i][j] = 1

        return grid

    return None


# Main execution
# image_path = '/home/infcapital/Documents/yolov8/p3/p3_filled.png'
# parts = split_image(image_path)
#
# for i, part in enumerate(parts, 1):
#     print(f"\nXử lý phần {i}:")
#     grid = detect_circles_and_create_grid(part)
#     if grid is not None:
#         print(f"Ma trận cho phần {i}:")
#         print(grid)
#     else:
#         print(f"Không thể nhận diện hình tròn hoặc tạo lưới cho phần {i}.")
#
# print("\nĐã hoàn thành việc xử lý tất cả các phần.")