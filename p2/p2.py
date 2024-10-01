import cv2
import numpy as np
import matplotlib.pyplot as plt


def remove_text(image):
    # Convert to grayscale if not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Apply threshold
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask
    mask = np.ones(gray.shape, dtype=np.uint8) * 255

    for contour in contours:
        # Calculate contour properties
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)

        # If the contour is not circular enough, remove it
        if circularity < 0.7 or area < 100:  # Adjust these thresholds as needed
            cv2.drawContours(mask, [contour], 0, 0, -1)

    # Apply the mask to the original image
    result = cv2.bitwise_and(gray, mask)

    return result


def detect_circles_and_marks(image):
    # Remove text
    cleaned_image = remove_text(image)

    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(cleaned_image, (5, 5), 0)

    # Use HoughCircles to detect all circles
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=30,
        param1=50,
        param2=30,
        minRadius=15,
        maxRadius=30
    )

    marked_circles = []
    unmarked_circles = []

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            x, y, r = circle
            # Check if the circle is marked (has significant black pixels inside)
            mask = np.zeros(cleaned_image.shape, np.uint8)
            cv2.circle(mask, (x, y), r - 2, 255, -1)
            mean_val = cv2.mean(cleaned_image, mask=mask)[0]

            if mean_val < 200:  # Adjust this threshold as needed
                marked_circles.append(circle)
            else:
                unmarked_circles.append(circle)

    return marked_circles, unmarked_circles, cleaned_image


def create_grid(image, marked_circles, unmarked_circles):
    height, width = image.shape[:2]
    all_circles = marked_circles + unmarked_circles

    # Sort circles by y-coordinate (row)
    sorted_circles = sorted(all_circles, key=lambda c: c[1])

    # Identify unique rows
    rows = []
    current_row = []
    for circle in sorted_circles:
        if not current_row or abs(circle[1] - current_row[0][1]) < height * 0.05:  # 5% của chiều cao
            current_row.append(circle)
        else:
            rows.append(sorted(current_row, key=lambda c: c[0]))  # Sắp xếp theo x trong mỗi hàng
            current_row = [circle]
    if current_row:
        rows.append(sorted(current_row, key=lambda c: c[0]))

    return rows


def create_answer_matrix(rows, marked_circles):
    answer_matrix = []
    for row in rows:
        row_answers = []
        for circle in row:
            if any(np.array_equal(circle, mc) for mc in marked_circles):
                row_answers.append(1)
            else:
                row_answers.append(0)
        answer_matrix.append(row_answers)
    return answer_matrix


def process_answer_sheet_p2(image_path):
    # Read the image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect marked and unmarked circles
    marked_circles, unmarked_circles, cleaned_image = detect_circles_and_marks(image)

    # Create grid
    rows = create_grid(image, marked_circles, unmarked_circles)

    # Create answer matrix
    answer_matrix = create_answer_matrix(rows, marked_circles)

    # Create result image
    result_img = image_rgb.copy()

    # Draw grid and circles
    for i, row in enumerate(rows):
        for j, circle in enumerate(row):
            color = (0, 255, 0) if answer_matrix[i][j] == 1 else (255, 0, 0)
            cv2.circle(result_img, (circle[0], circle[1]), circle[2], color, 2)
            cv2.putText(result_img, f"{i + 1},{j + 1}", (circle[0] - 20, circle[1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Display results
    # plt.figure(figsize=(18, 12))
    # plt.subplot(221), plt.imshow(image_rgb), plt.title('Original Image')
    # plt.subplot(222), plt.imshow(cleaned_image, cmap='gray'), plt.title('Cleaned Image')
    # plt.subplot(223), plt.imshow(result_img), plt.title('Detected Grid and Answers')
    # plt.subplot(224), plt.imshow(answer_matrix, cmap='binary'), plt.title('Answer Matrix')
    # for i in range(len(answer_matrix)):
    #     for j in range(len(answer_matrix[i])):
    #         plt.text(j, i, str(answer_matrix[i][j]), ha='center', va='center')
    # plt.tight_layout()
    # plt.show()

    return answer_matrix



# Use the function
# image_path = 'p3/p3.png'  # Replace with your image path
# answer_matrix = process_answer_sheet(image_path)
# print("Answer Matrix:")
# for row in answer_matrix:
#     print(row)

