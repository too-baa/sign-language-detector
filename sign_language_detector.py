import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import tkinter as tk
from PIL import Image, ImageTk

# Ensure 'hand_landmarker.task' is in the same folder as this script
model_path = 'hand_landmarker.task'

class SignLanguageDetector:
    def __init__(self):
        # Configure MediaPipe Tasks Hand Landmarker
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.cap = cv2.VideoCapture(0)
        self.current_label = "Waiting for hand..."

    def process_frame(self, frame):
        # Convert BGR (OpenCV) to RGB (MediaPipe requirement)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        # Detect hands
        timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
        results = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        
        # Drawing landmarks
        if results.hand_landmarks:
            self.current_label = "Hand Detected"
            for hand_landmarks in results.hand_landmarks:
                for lm in hand_landmarks:
                    h, w, _ = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
        else:
            self.current_label = "No Hand Detected"
            
        return frame

class AppGUI:
    def __init__(self, root):
        self.detector = SignLanguageDetector()
        self.root = root
        self.root.title("Sign Language Detector")
        self.root.geometry("800x650")
        
        self.video_label = tk.Label(root)
        self.video_label.pack()
        
        self.status_label = tk.Label(root, text="Status: Starting...", font=("Arial", 16))
        self.status_label.pack(pady=10)
        
        self.update_video()

    def update_video(self):
        ret, frame = self.detector.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            frame = self.detector.process_frame(frame)
            
            # Display in GUI
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
            self.status_label.configure(text=f"Status: {self.detector.current_label}")
        
        self.root.after(10, self.update_video)

if __name__ == "__main__":
    root = tk.Tk()
    app = AppGUI(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.detector.cap.release(), root.destroy()))
    root.mainloop()