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
    # show_image(img, "1. Ảnh gốc")

    # Chuyển đổi sang ảnh grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # show_image(gray, "2. Ảnh grayscale")

    # Áp dụng ngưỡng để tạo ảnh nhị phân
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    # show_image(binary, "3. Ảnh nhị phân")

    # Tìm các đường kẻ dọc
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, binary.shape[0] // 2))
    detect_vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    # show_image(detect_vertical, "4. Phát hiện đường kẻ dọc")

    # Tìm các đường thẳng
    lines = cv2.HoughLinesP(detect_vertical, 1, np.pi / 180, threshold=100, minLineLength=binary.shape[0] // 2,
                            maxLineGap=20)

    # Vẽ các đường thẳng lên ảnh gốc
    img_with_lines = img.copy()
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img_with_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # show_image(img_with_lines, "5. Ảnh với các đường thẳng được phát hiện")

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


def detect_circles_and_draw_matrix(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Cắt bớt phần trên, trái và dưới của ảnh để loại bỏ text
    height, width = gray.shape
    gray = gray[int(height * 0.17):int(height * 0.97), int(width * 0.1):]
    img = img[int(height * 0.17):int(height * 0.97), int(width * 0.1):]

    # Áp dụng ngưỡng
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # Tìm contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Lọc và lưu trữ tâm các đường tròn
    centers = []
    filled_centers = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if 20 < area < 300:  # Điều chỉnh ngưỡng diện tích nếu cần
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                # Kiểm tra xem đường tròn có được tô hay không
                mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.drawContours(mask, [contour], 0, 255, -1)
                mean_val = cv2.mean(gray, mask=mask)[0]

                if mean_val < 100:  # Ngưỡng để xác định đường tròn được tô
                    filled_centers.append((cX, cY))
                else:
                    centers.append((cX, cY))

    # Sắp xếp các tâm theo tọa độ y tăng dần
    all_centers = centers + filled_centers

    # Check if any circles were detected
    if not all_centers:
        print("No circles detected in the image.")
        return np.zeros((1, 1), dtype=int)  # Return an empty matrix

    all_centers.sort(key=lambda c: c[1])

    # Phát hiện các hàng dựa trên khoảng cách y
    rows = []
    current_row = [all_centers[0]]
    for center in all_centers[1:]:
        if abs(center[1] - current_row[-1][1]) > 20:  # Ngưỡng khoảng cách giữa các hàng
            rows.append(current_row)
            current_row = [center]
        else:
            current_row.append(center)
    rows.append(current_row)

    # Sắp xếp các điểm trong mỗi hàng theo tọa độ x
    for row in rows:
        row.sort(key=lambda c: c[0])

    # Tìm số cột tối đa
    max_cols = max(len(row) for row in rows)

    # Tạo ma trận biểu diễn
    matrix = np.zeros((len(rows), max_cols), dtype=int)
    for i, row in enumerate(rows):
        for j, (cX, cY) in enumerate(row):
            if (cX, cY) in filled_centers:
                matrix[i, j] = 1

    # Vẽ các đường tròn đã phát hiện
    for (cX, cY) in centers:
        cv2.circle(img, (cX, cY), 10, (0, 255, 0), 2)
    for (cX, cY) in filled_centers:
        cv2.circle(img, (cX, cY), 10, (0, 0, 255), -1)

    return matrix



