import cv2

from detect_circle import process_answer_sheet_p1
from p2.p2 import process_answer_sheet_p2
from p3.p3 import split_image, detect_circles_and_draw_matrix
from p3.split_image import detect_circles_and_create_grid

MATRIX_ANSWER_P1=[[[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1],
                     [1, 0, 0, 0],
                     [0, 0, 1, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1],
                     [1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 1, 0, 0]],[[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1],
                     [1, 0, 0, 0],
                     [0, 0, 1, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1],
                     [1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 1, 0, 0]],[[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1],
                     [1, 0, 0, 0],
                     [0, 0, 1, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1],
                     [1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 1, 0, 0]],[[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1],
                     [1, 0, 0, 0],
                     [0, 0, 1, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1],
                     [1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 1, 0, 0]]]

MATRIX_ANSWER_P2=[[[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1],
                     [1, 0, 0, 0]],
                     [[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1],
                     [1, 0, 0, 0]],
                     [[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1],
                     [1, 0, 0, 0]],
                     [[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1],
                     [1, 0, 0, 0]]]

MATRIX_ANSWER_P3=[[[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1],
                     [1, 0, 0, 0],
                     [0, 0, 1, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1],
                     [1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 1, 0, 0],
                     [0, 1, 0, 0],
                     [0, 1, 0, 0]
                   ],[[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1],
                     [1, 0, 0, 0],
                     [0, 0, 1, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1],
                     [1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 1, 0, 0],
                     [0, 1, 0, 0],
                     [0, 1, 0, 0]],[[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1],
                     [1, 0, 0, 0],
                     [0, 0, 1, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1],
                     [1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 1, 0, 0],
                     [0, 1, 0, 0],
                     [0, 1, 0, 0]],[[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1],
                     [1, 0, 0, 0],
                     [0, 0, 1, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1],
                     [1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 1, 0, 0],
                     [0, 1, 0, 0],
                     [0, 1, 0, 0]],[[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1],
                     [1, 0, 0, 0],
                     [0, 0, 1, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1],
                     [1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 1, 0, 0],
                     [0, 1, 0, 0],
                     [0, 1, 0, 0]],[[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1],
                     [1, 0, 0, 0],
                     [0, 0, 1, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1],
                     [1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 1, 0, 0],
                     [0, 1, 0, 0],
                     [0, 1, 0, 0]]]

def resize_image(image, width=800):
    aspect_ratio = float(image.shape[1]) / float(image.shape[0])
    height = int(width / aspect_ratio)
    return cv2.resize(image, (width, height))

def compare_answer_matrices(matrix1, matrix2):
    score = 0
    for i in range(len(matrix1)):
        for j in range(len(matrix1[i])):
            if matrix1[i][j] == matrix2[i][j] and matrix1[i][j] == 1:
                score += 1
    return score

def compare_answers(matrix_answers, matrix_answers_true):
    sum_score = 0
    for i in range(len(matrix_answers)):
        score = compare_answer_matrices(matrix_answers[i],matrix_answers_true[i])
        sum_score += score
    return sum_score

def score_exam(exam_image_path):
    if exam_image_path is None:
        print("Error: Could not read one or both input images.")
        return
    empty_matrix = process_answer_sheet_p1(exam_image_path)
    print(empty_matrix)
    filled_matrix = [[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1],
                     [1, 0, 0, 0],
                     [0, 0, 1, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1],
                     [1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 1, 0, 0]]
    print("Empty Sheet Matrix:")

    score = compare_answer_matrices(empty_matrix, filled_matrix)
    return score

def detect_and_extract_cells(image_path):
    # Load image, grayscale, adaptive threshold
    image = cv2.imread(image_path)
    result = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 9)

    # Fill rectangular contours
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(thresh, [c], -1, (255,255,255), -1)

    # Morph open
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=4)

    # Draw rectangles and store their information
    cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    rectangles = []
    for i, c in enumerate(cnts, 1):
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 3)
        cv2.putText(image, str(i), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        rectangles.append((i, x, y, w, h))

    # Sort rectangles by their number
    rectangles.sort(key=lambda r: r[0])

    # Extract cells 3 and 12
    cells_to_extract = [3,4,5,6,7,8,9,10,11]
    total_score = 0
    matrixs_p1 = []
    matrixs_p2=[]
    for cell_num in cells_to_extract:
        if cell_num <= len(rectangles):
            i, x, y, w, h = rectangles[cell_num - 1]

            if 8 <= i <= 11:
                cell = result[y:y+h, x:x+w]
                filename_path = f'/home/infcapital/Documents/yolov8/data/cell_{i}.jpg'
                cv2.imwrite(filename_path, cell)
                matrix_p1 = process_answer_sheet_p1(filename_path)
                matrixs_p1.append(matrix_p1)
            elif 4 <= i <= 7:
                cell = result[y:y + h, x:x + w]
                filename_path = f'/home/infcapital/Documents/yolov8/data/cell_{i}.jpg'
                cv2.imwrite(filename_path, cell)
                matrix_p2 = process_answer_sheet_p2(filename_path)
                matrixs_p2.append(matrix_p2)

            elif i == 3:
                cell = result[y:y + h, x:x + w]
                filename_path = f'/home/infcapital/Documents/yolov8/data/cell_{i}.jpg'
                cv2.imwrite(filename_path, cell)
                parts = split_image(filename_path)
                matrixs = []
                for i, part in enumerate(parts, 1):
                    matrix_p3 = detect_circles_and_create_grid(part)
                    # print(matrix_p3)
                    matrixs.append(matrix_p3)
                # print('matrixs',matrixs)
                score = compare_answers(matrixs, MATRIX_ANSWER_P3)
                total_score += score
    if len(matrixs_p1) > 0:
        score = compare_answers(matrixs_p1, MATRIX_ANSWER_P1)
        total_score += score
    if len(matrixs_p2) > 0:
        score = compare_answers(matrixs_p2, MATRIX_ANSWER_P2)
        total_score += score
    print('Score : ', total_score)


if __name__ == '__main__':
    detect_and_extract_cells('/home/infcapital/Documents/yolov8/image_filled.png')