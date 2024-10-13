import cv2
import time
from collections import defaultdict
from ultralytics import YOLO

class SpeedCalculator:
    def __init__(self, focal_length, real_width):
        self.focal_length = focal_length  # Focal length of the camera
        self.real_width = real_width  # Real width of the car (in meters)

    def calculate_speed(self, distance1, distance2, time_elapsed):
        distance_covered = abs(distance2 - distance1)  # Ensure positive speed
        speed = distance_covered / time_elapsed
        return speed * 3.6  # Convert m/s to km/h

    def calculate_distance(self, object_width):
        distance = (self.real_width * self.focal_length) / object_width
        return distance

def detect_vehicles_yolo(frame, model):
    height, width = frame.shape[:2]
    results = model(frame)
    detections = []

    for result in results[0].boxes:
        xyxy = result.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = xyxy[0], xyxy[1], xyxy[2], xyxy[3]
        confidence = result.conf[0].cpu().numpy()
        class_id = int(result.cls[0].cpu().numpy())

        if confidence > 0.5 and class_id == 2:  # Filter for cars only (class_id 2 for cars in COCO)
            w = x2 - x1
            h = y2 - y1
            center_x = (x1 + x2) / 2
            if is_in_lane(center_x, width):
                detections.append((int(x1), int(y1), int(w), int(h)))

    return detections

def is_in_lane(center_x, frame_width):
    lane_width = frame_width / 3
    return lane_width < center_x < 2 * lane_width

def control_speed(distance):
    if distance < 10:
        print("Emergency brake")
    elif 10 <= distance < 20:
        print("Reducing speed")
    elif 20 <= distance < 40:
        print("Maintaining speed")
    elif 40 <= distance < 60:
        print("Slightly increasing speed")
    else:
        print("Increasing speed")

def main():
    cap = cv2.VideoCapture("testL.mp4")
    focal_length = 700  # Example focal length in pixels
    real_vehicle_width = 1.8  # Average width of a vehicle in meters

    speed_calculator = SpeedCalculator(focal_length, real_vehicle_width)
    model = YOLO('yolov10n.pt')

    previous_distances = defaultdict(lambda: None)
    previous_times = defaultdict(lambda: None)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detections = detect_vehicles_yolo(frame, model)
        current_time = time.time()

        for idx, (x, y, w, h) in enumerate(detections):
            object_width = w
            current_distance = speed_calculator.calculate_distance(object_width)

            if previous_distances[idx] is not None and previous_times[idx] is not None:
                time_elapsed = current_time - previous_times[idx]
                if time_elapsed > 0:
                    speed = speed_calculator.calculate_speed(previous_distances[idx], current_distance, time_elapsed)
                    cv2.putText(frame, f"Speed: {speed:.2f} km/h", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    control_speed(current_distance)

            previous_distances[idx] = current_distance
            previous_times[idx] = current_time

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Distance: {current_distance:.2f} m", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('Adaptive Cruise Control', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
