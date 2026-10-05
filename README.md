# Real-Time Sign Language Translation System 🤟

An AI-powered, real-time sign language detection and translation system built with Python, OpenCV, MediaPipe, and Scikit-Learn. This project was developed as part of a social internship initiative to provide an accessible, low-cost assistive technology tool for the speech- and hearing-impaired community.

---

## 🚀 Features
* **Real-Time Hand Tracking:** Utilizes MediaPipe to track 21 3D hand landmarks at 30+ frames per second via a standard webcam.
* **Position-Independent Recognition:** Implements wrist-coordinate normalization so hand signs are accurately recognized regardless of where the hand is positioned on the screen.
* **Machine Learning Classification:** Powered by a robust Random Forest Classifier (`scikit-learn`) trained on custom gesture datasets.
* **Interactive GUI:** Features a clean desktop interface built with Tkinter for live video streaming and instant text prediction.

---

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Computer Vision:** OpenCV (`cv2`)
* **Hand Tracking:** MediaPipe (Legacy Solutions / Tasks API)
* **Machine Learning:** Scikit-Learn (Random Forest)
* **GUI Framework:** Tkinter, Pillow (PIL)
* **Data Handling:** Pandas, NumPy, Joblib

---

## 📁 Project Structure
```text
├── main.py                 # Main GUI application & real-time prediction loop
├── data_collector.py       # Script to record and save hand coordinate data
├── train.py                # Script to train the Random Forest model
├── dataset.csv             # Spreadsheet containing recorded hand landmarks
├── sign_model.pkl          # Saved machine learning model ("brain")
└── README.md               # Project documentation
```

---

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/sign-language-translator.git
   cd sign-language-translator
   ```

2. **Install required dependencies:**
   ```bash
   pip install opencv-python mediapipe scikit-learn pandas numpy joblib pillow
   ```

---

## 🕹️ How to Run the Project

### Step 1: Collect Custom Data (Optional if dataset exists)
If you want to record your own hand signs:
```bash
python data_collector.py
```
*(Enter the name of the letter/sign when prompted, position your hand in front of the webcam, and collect samples).*

### Step 2: Train the Model
Compile your dataset into a trained model "brain":
```bash
python train.py
```
*(This generates or updates `sign_model.pkl`).*

### Step 3: Run the Main Application
Launch the real-time translation GUI:
```bash
python main.py
```

---

## 🔍 Troubleshooting (Stuck on Predicting a Single Letter?)
If your model gets stuck predicting only one letter (like "T" or "A"):
1. **Did you re-train?** Make sure you run `python train.py` *after* updating or re-collecting your `dataset.csv`.
2. **Check your dataset:** Open `dataset.csv` in Excel or Notepad to ensure it contains multiple distinct classes/labels in the first column and isn't dominated by a single letter.
3. **Delete old cache:** Delete `sign_model.pkl` and re-run `train.py` to ensure a fresh model is loaded.

---

## 🌍 Social Impact & Future Scope
* **Impact:** Designed to bridge communication gaps in classrooms, public kiosks, and daily interactions for hearing-impaired individuals.
* **Future Scope:** Expanding the system from individual letter recognition to full-sentence translation and integrating mobile/web platforms.

---

## 📝 License
This project is open-source and available under the [MIT License](LICENSE).