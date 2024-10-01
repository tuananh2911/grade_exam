import cv2
import numpy as np

from p1.detect_circle import process_answer_sheet


def resize_image(image, width=800):
    aspect_ratio = float(image.shape[1]) / float(image.shape[0])
    height = int(width / aspect_ratio)
    return cv2.resize(image, (width, height))


def generate_answer_matrix(image):
    result = process_answer_sheet(image)
    return result


def compare_answer_matrices(matrix1, matrix2):
    score = 0
    for i in range(len(matrix1)):
        for j in range(len(matrix1[i])):
            if matrix1[i][j] == matrix2[i][j] and matrix1[i][j] == 1:
                score += 1
    return score



