import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import csv
import os

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
landmarker = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
label = input("Enter the label/letter you are recording: ")
data = []
samples_collected = 0

print(f"Collecting data for '{label}'. Press 'q' to stop.")
while samples_collected < 100:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    
    results = landmarker.detect(mp_image)
    
    if results.hand_landmarks:
        # Get wrist (landmark 0) to normalize
        wrist = results.hand_landmarks[0][0]
        landmarks_flat = []
        for lm in results.hand_landmarks[0]:
            # Save distance from wrist instead of absolute screen position
            landmarks_flat.extend([lm.x - wrist.x, lm.y - wrist.y])
        
        data.append([label] + landmarks_flat)
        samples_collected += 1
        cv2.putText(frame, f"Collected: {samples_collected}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Collecting Data', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

with open('dataset.csv', 'a', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(data)

cap.release()
cv2.destroyAllWindows()
print("Data saved!")