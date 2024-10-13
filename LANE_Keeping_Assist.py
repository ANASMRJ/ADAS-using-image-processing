import cv2 as cv
import numpy as np
from math import atan, degrees
import json

def canny(img):
    if img is None:
        cap.release()
        cv.destroyAllWindows()
        exit()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    canny_edges = cv.Canny(blur, 50, 150)
    return canny_edges
"""
Cette fonction applique l'algorithme de détection de contours Canny sur une image.
Elle convertit d'abord l'image en niveaux de gris, applique un flou gaussien pour réduire le bruit,
puis utilise cv.Canny pour détecter les contours.
"""
def red_white_masking(image):
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    lower_y, upper_y = np.array([10, 130, 120], np.uint8), np.array([40, 255, 255], np.uint8)
    lower_w, upper_w = np.array([0, 0, 212], np.uint8), np.array([170, 200, 255], np.uint8)
    mask_y, mask_w = cv.inRange(hsv, lower_y, upper_y), cv.inRange(hsv, lower_w, upper_w)
    mask = cv.bitwise_or(mask_w, mask_y)
    return cv.bitwise_and(image, image, mask=mask)
"""
Cette fonction crée un masque pour détecter les couleurs rouge et blanche sur une image.
L'image est convertie en espace de couleur HSV, puis des seuils de couleur sont appliqués pour créer des masques pour les couleurs rouges et blanches.
Ces masques sont combinés et appliqués à l'image d'origine.
"""
def apply_filter(image):
    kernel = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    return cv.filter2D(image, -1, kernel)
"""
Cette fonction applique un filtre de détection de bord à l'image en utilisant un noyau défini pour détecter les bords verticaux.
"""
def region_of_interest(image, vertices):
    mask = np.zeros_like(image)
    cv.fillConvexPoly(mask, cv.convexHull(vertices), 255)
    return cv.bitwise_and(image, mask)
"""
Cette fonction applique un masque polygonal sur l'image pour ne conserver que la région d'intérêt spécifiée par les vertices (sommets).
Les pixels en dehors de cette région sont masqués.
"""
def detect_edges(image):
    return cv.Canny(image, 80, 200)
"""
Cette fonction utilise l'algorithme de détection de contours Canny avec des seuils spécifiques pour détecter les bords dans une image.
"""
def average_slope_intercept(image, lines):
    left_fit, right_fit = [], []
    global left_line_params, right_line_params
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        slope, intercept = np.polyfit((x1, x2), (y1, y2), 1)
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis=0) if left_fit else left_line_params
    right_fit_average = np.average(right_fit, axis=0) if right_fit else right_line_params
    left_line_params, right_line_params = left_fit_average, right_fit_average
    return np.array([create_coordinates(image, left_fit_average), create_coordinates(image, right_fit_average)])
"""
Cette fonction calcule la pente moyenne et l'interception des lignes détectées pour les lignes de gauche et de droite.
Elle utilise la régression linéaire pour trouver les lignes de meilleure adaptation et retourne les coordonnées des lignes.
"""
def create_coordinates(image, line_params):
    slope, intercept = line_params
    y1, y2 = image.shape[0], int(image.shape[0] * (2 / 3))
    x1, x2 = int((y1 - intercept) / slope), int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])
"""
Cette fonction génère les coordonnées des points de début et de fin d'une ligne basée sur les paramètres de la ligne (pente et intercept).
Définit y1 comme la hauteur de l'image et y2 comme les deux tiers de la hauteur de l'image.
Calcule x1 et x2 en utilisant l'équation de la ligne y=mx+c (où m est la pente et c est l'intercept).
Retourne un tableau avec les coordonnées [x1, y1, x2, y2].
"""
def draw_lane_lines(image, left_line, right_line, danger_side):
    color_left = (0, 0, 255) if danger_side == 'L' else (0, 255, 0)
    color_right = (0, 0, 255) if danger_side == 'R' else (0, 255, 0)
    cv.line(image, (left_line[0], left_line[1]), (left_line[2], left_line[3]), color_left, 5)
    cv.line(image, (right_line[0], right_line[1]), (right_line[2], right_line[3]), color_right, 5)
    vertices = np.array([[(left_line[0], left_line[1]), (left_line[2], left_line[3]), 
                          (right_line[2], right_line[3]), (right_line[0], right_line[1])]], np.int32)
    mask = np.zeros_like(image)
    cv.fillConvexPoly(mask, vertices[0], (200, 0, 0))
    return cv.addWeighted(image, 1, mask, 0.5, 0)
"""
Cette fonction dessine les lignes de la voie sur l'image.
Détermine la couleur des lignes en fonction du côté dangereux (danger_side).
Dessine les lignes gauche et droite avec les couleurs respectives.
Crée un polygone couvrant la région entre les deux lignes et le remplit avec une couleur semi-transparente.
Superpose ce polygone sur l'image originale en utilisant cv.addWeighted.
"""
def process_frame(image):
    height, width = image.shape[:2]
    masked_image = red_white_masking(image)
    filtered_image = apply_filter(masked_image)
    gray_image = cv.cvtColor(filtered_image, cv.COLOR_BGR2GRAY)
    vertices = np.array([[(0, height), (width / 2, height / 2), (width, height)]], np.int32)
    roi_image = region_of_interest(gray_image, vertices)
    edges = detect_edges(roi_image)
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, 20, minLineLength=5, maxLineGap=200)
    left_lane, right_lane = average_slope_intercept(image, lines)
    center_of_lanes = (left_lane + right_lane) / 2
    center_of_frame = width // 2
    deviation = center_of_frame - center_of_lanes[0]
    angle_degrees = degrees(atan(deviation / center_of_frame))
    direction = "R" if deviation < 0 else "L"
    danger_side = direction if abs(angle_degrees) > 10 else None
    final_image = draw_lane_lines(image.copy(), left_lane, right_lane, danger_side)
    cv.putText(final_image, f"Move {direction} ({abs(angle_degrees):.2f} degrees)", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
    if abs(angle_degrees) > 10:
        cv.putText(final_image, "Warning!!! Car out of Lane", (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
    print(f"Move {direction} with a deviation of {abs(angle_degrees):.2f} degrees")
    print(json.dumps({"r": direction, "b": abs(angle_degrees)}))
    return final_image
"""
Cette fonction traite une image de cadre de vidéo.
"""
cap = cv.VideoCapture("testL.mp4")
left_line_params, right_line_params = None, None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        cap.set(cv.CAP_PROP_POS_FRAMES, 0)
        continue
    resized_frame = cv.resize(frame, (1080, 640))
    detected_frame = process_frame(resized_frame)
    cv.imshow('Lane keeping assist', detected_frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cv.destroyAllWindows()
cap.release()