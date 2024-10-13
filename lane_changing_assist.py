import cv2
from ultralytics import YOLO

# Chemin vers la vidéo
video_path = 'testL.mp4'

# Charger les classes d'objets à partir d'un fichier texte
# Le fichier 'coco.txt' contient les noms des classes du dataset COCO
with open('coco.txt', 'r') as f:
    classes = f.read().splitlines()

# Charger le modèle YOLOv10n
model = YOLO('yolov10n.pt')

def detect_adjacent_vehicles_yolo(frame):
    height, width, _ = frame.shape
    
    # Passer l'image dans le modèle YOLOv10n pour obtenir les détections
    results = model(frame)
    
    # Initialiser les indicateurs de détection pour les voies gauche et droite
    left_detected = False
    right_detected = False
    
    # Parcourir les détections
    for result in results[0].boxes:
        xyxy = result.xyxy[0]  # Accéder aux coordonnées (x1, y1, x2, y2)
        x1, y1, x2, y2 = xyxy[0], xyxy[1], xyxy[2], xyxy[3]
        conf = result.conf[0]
        class_id = int(result.cls[0])
        
        # Filtrer uniquement les détections de véhicules (classe 'car' dans le dataset COCO)
        if conf > 0.5 and class_id == 2:  # Classe 'car' dans COCO dataset
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            if center_x < width // 2:
                left_detected = True
                # Dessiner un cercle pour marquer la détection dans la voie de gauche
                cv2.circle(frame, (int(center_x), int(center_y)), 10, (0, 255, 0), 2)
            elif center_x > width // 2:
                right_detected = True
                # Dessiner un cercle pour marquer la détection dans la voie de droite
                cv2.circle(frame, (int(center_x), int(center_y)), 10, (0, 255, 0), 2)
    
    return left_detected, right_detected
"""
    Fonction pour détecter les véhicules dans les voies adjacentes en utilisant YOLOv10n.
    Arguments:
    - frame: image à traiter pour la détection.

    Retourne:
    - left_detected: booléen indiquant si un véhicule est détecté dans la voie de gauche.
    - right_detected: booléen indiquant si un véhicule est détecté dans la voie de droite.
"""

def assist_lane_change(frame):
    # Détecter les véhicules dans les voies adjacentes
    left_detected, right_detected = detect_adjacent_vehicles_yolo(frame)
    
    # Définir les messages d'avertissement
    if left_detected and right_detected:
        message = "Attention: Vehicules detectes dans les deux voies adjacentes !"
    elif left_detected:
        message = "Attention: Vehicule detecte dans la voie de droite."
    elif right_detected:
        message = "Attention: Vehicule detecte dans la voie de gauche."
    else:
        message = "Changement de voie securitaire."

    # Afficher le message sur le frame
    cv2.putText(frame, message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    """
    Fonction pour simuler l'assistance au changement de voie en affichant des avertissements.
    Arguments:
    - frame: image à traiter pour la détection.

    La fonction affiche des messages d'avertissement sur l'image en fonction des résultats
    de la détection des véhicules dans les voies adjacentes.
    """

# Lecture de la vidéo
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Simuler l'assistance au changement de voie pour chaque frame de la vidéo
    assist_lane_change(frame)
    
    # Afficher la frame (facultatif, à des fins de débogage)
    cv2.imshow('Lane Changing Assist', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer la capture vidéo et fermer les fenêtres OpenCV
cap.release()
cv2.destroyAllWindows()
