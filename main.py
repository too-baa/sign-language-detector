import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import tkinter as tk
from PIL import Image, ImageTk
import joblib
import numpy as np

# Load your trained model
model = joblib.load('sign_model.pkl')

class SignLanguageDetector:
    def __init__(self):
        # Configure MediaPipe Tasks
        base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.cap = cv2.VideoCapture(0)
        self.current_label = "Waiting for hand..."

    def process_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
        results = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        
        if results.hand_landmarks:
            # Extract landmarks for the model
            landmarks_flat = []
            for lm in results.hand_landmarks[0]:
                landmarks_flat.extend([lm.x, lm.y])
            
            # Predict using the loaded model
            prediction = model.predict([landmarks_flat])
            self.current_label = f"Predicted: {prediction[0]}"
            
            # Draw landmarks
            for lm in results.hand_landmarks[0]:
                h, w, _ = frame.shape
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 5, (0, 255, 0), -1)
        else:
            self.current_label = "No Hand Detected"
            
        return frame

class AppGUI:
    def __init__(self, root):
        self.detector = SignLanguageDetector()
        self.root = root
        self.root.title("Sign Language Translator")
        self.root.geometry("800x650")
        self.video_label = tk.Label(root)
        self.video_label.pack()
        self.status_label = tk.Label(root, text="Status: Starting...", font=("Arial", 20, "bold"))
        self.status_label.pack(pady=20)
        self.update_video()

    def update_video(self):
        ret, frame = self.detector.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            frame = self.detector.process_frame(frame)
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
            self.status_label.configure(text=self.detector.current_label)
        self.root.after(10, self.update_video)

if __name__ == "__main__":
    root = tk.Tk()
    app = AppGUI(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.detector.cap.release(), root.destroy()))
    root.mainloop()