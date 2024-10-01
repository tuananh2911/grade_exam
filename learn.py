import cv2


def resize_image(image, width=800):
    aspect_ratio = float(image.shape[1]) / float(image.shape[0])
    height = int(width / aspect_ratio)
    return cv2.resize(image, (width, height))


def show_image(title, image):
    cv2.imshow(title, resize_image(image))
    cv2.waitKey(0)


def detect_and_extract_cells():
    # Load image, grayscale, adaptive threshold
    image = cv2.imread('/home/infcapital/Documents/yolov8/cell_3.jpg')
    show_image("Original Image", image)

    result = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    show_image("Grayscale Image", gray)

    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 9)
    show_image("Thresholded Image", thresh)

    # Fill rectangular contours
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(thresh, [c], -1, (255, 255, 255), -1)
    show_image("Filled Contours", thresh)

    # Morph open
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=4)
    show_image("Morphological Opening", opening)

    # Draw rectangles and store their information
    cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    rectangles = []
    for i, c in enumerate(cnts, 1):
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 3)
        cv2.putText(image, str(i), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        rectangles.append((i, x, y, w, h))
    show_image("Detected Cells", image)

    # Sort rectangles by their number
    rectangles.sort(key=lambda r: r[0])

    # Extract cells 3 and 12
    cells_to_extract = [3]
    total_score = 0
    for cell_num in cells_to_extract:
        if cell_num <= len(rectangles):
            i, x, y, w, h = rectangles[cell_num - 1]
            print('i', i)
            if i == 3:
                cell = result[y:y + h, x:x + w]
                filename_path = f'cell_{i}.jpg'
                cv2.imwrite(filename_path, cell)
                show_image(f"Extracted Cell {i}", cell)
                # score = score_exam(filename_path)
                # print('score per image', score)
                # total_score += score


    print('Score : ', total_score)


if __name__ == '__main__':
    detect_and_extract_cells()
    cv2.destroyAllWindows()