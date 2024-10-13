# ADAS-using-image-processing
I developed three Python codes for ADAS functionalities: Lane Keeping Assist (LKA) to maintain lane position, Lane Change Assist (LCA) for safe lane transitions, and Adaptive Cruise Control (ACC) for automatic speed adjustments.

________________________________________________________________________________________________________________________________________
**Lane Keeping Assist (LKA) Code Description**
![image](https://github.com/user-attachments/assets/339badc2-cafd-40c6-a5b4-56461ca3a8ed)

This Python script utilizes OpenCV to implement a Lane Keeping Assist (LKA) system. It processes video frames to detect lane markings and calculate the vehicle's deviation from the lane center. The primary functions include:

Canny Edge Detection: Applies the Canny edge detection algorithm to identify edges in the image, crucial for lane detection.

Color Masking: Uses HSV color space to create masks for red and white colors, enhancing the visibility of lane markings.

Region of Interest: Focuses the analysis on a specific area of the frame where lanes are expected to be found.

Line Detection: Uses Hough Transform to detect lines in the processed image.

Slope and Intercept Calculation: Computes the average slope and intercept of detected lines to establish left and right lane boundaries.

Lane Visualization: Draws the detected lane lines on the original image and indicates which direction (left or right) the vehicle should move to stay in the lane based on the calculated deviation from the center.

Warnings and Feedback: Provides real-time feedback through on-screen text, indicating the vehicle's deviation angle and issuing warnings if the vehicle is drifting out of the lane.

The program captures video from a file, processes each frame to detect lanes, and displays the results in real-time until the user decides to quit.

________________________________________________________________________________________________________________________________________
**Lane Change Assist (LCA) Code Description**
![image](https://github.com/user-attachments/assets/1b906854-f697-431a-93be-76cda4feb342)
![image](https://github.com/user-attachments/assets/4b29f7d5-4b50-4dc3-a755-cd8e4eaf0588)


This Python script utilizes the YOLOv10 object detection model from the Ultralytics library to assist drivers in safely changing lanes by detecting adjacent vehicles. The main components of the code are as follows:

Video Input: The script begins by loading a video file (Robot.mp4) for processing.

Class Loading: It reads class labels from a coco.txt file, which contains the names of the object classes used in the COCO dataset.

Model Initialization: The YOLOv10 model is loaded using a pre-trained weights file (yolov10n.pt).

Vehicle Detection: The detect_adjacent_vehicles_yolo function processes each video frame:

It applies the YOLO model to detect objects in the frame.
It identifies vehicles (specifically cars, class ID 2) based on confidence thresholds.
For detected vehicles, it checks their positions relative to the frame's center and marks them with circles on the left or right side.
Lane Change Assistance: The assist_lane_change function utilizes the detection results to generate warning messages:

It checks if vehicles are detected in adjacent lanes and sets appropriate warning messages.
These messages are displayed on the video frame using OpenCV's putText function.
Video Processing Loop: The script continuously reads frames from the video, processes each frame for lane change assistance, and displays the results in a window. The loop runs until the video ends or the user decides to quit.

Cleanup: After processing, it releases the video capture and closes any OpenCV windows.

This implementation enhances driver awareness by providing real-time feedback on nearby vehicles during lane changes, contributing to safer driving experiences.
________________________________________________________________________________________________________________________________________
**Adaptive Cruise Control (ACC)**
Class Definition:
![image](https://github.com/user-attachments/assets/25e5ee5a-a20b-4076-839a-2cc0e9e6d23c)
![image](https://github.com/user-attachments/assets/2f788da8-b348-4331-95c3-f33aa679ed03)

SpeedCalculator: This class calculates the speed and distance of detected vehicles based on the focal length of the camera and the real width of the vehicle.
calculate_speed: Computes the speed of a vehicle in km/h using the distance covered and time elapsed.
calculate_distance: Calculates the distance to a vehicle based on its detected width.
Vehicle Detection:

detect_vehicles_yolo: Processes each frame to detect vehicles using the YOLO model.
It filters detections to only include cars (class ID 2 in COCO) and checks if they are in the specified lane using the is_in_lane function.
Detections are stored in a list, including bounding box coordinates and dimensions.
Lane Check:

is_in_lane: Checks if the detected vehicle's center position is within the lane bounds.
Speed Control:

control_speed: Issues commands based on the distance to the detected vehicle, simulating speed control actions (emergency brake, reducing speed, etc.).
Main Processing Loop:

The main function captures video from a file and initializes the YOLO model and speed calculator.
It continuously reads frames from the video, detects vehicles, calculates distances, and computes speeds.
The current speed and distance to each detected vehicle are displayed on the video frame.
Speed control actions are triggered based on the calculated distance.
