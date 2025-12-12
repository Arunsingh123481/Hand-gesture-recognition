🎯 Hand Gesture Recognition System

📌 Overview
A real-time hand gesture recognition system that uses MediaPipe for hand landmark detection and K-Nearest Neighbors (KNN) for gesture classification. This system can recognize multiple hand gestures in real-time using just a webcam!

✨ Features
🎯 Real-time Recognition: Process webcam feed at 30+ FPS

🤖 Machine Learning: KNN classifier for accurate gesture prediction

📊 Custom Training: Easy data collection and model training

🎭 Multiple Gestures: Supports OPEN, FIST, PEACE (extendable)

🖥️ Cross-platform: Works on Windows, Mac, and Linux

📈 Easy to Extend: Add your own custom gestures

🚀 Quick Start
1. Clone the Repository
bash
git clone https://github.com/Arunsingh123481/Hand-gesture-recognition.git
cd Hand-gesture-recognition
2. Install Dependencies
bash
pip install -r requirements.txt
If requirements.txt is not available:

bash
pip install opencv-python mediapipe scikit-learn pandas joblib numpy
3. Run the Recognition System
bash
python gesture_recognition.py
📁 Project Structure
text
Hand-gesture-recognition/
│
├── gesture_recognition.py          # Main recognition script
├── train_gesture_model.py          # Model training script
├── data_collector.py               # Data collection utility
├── gesture_data.csv               # Sample training dataset
├── realtime_gesture_recognition.py # Alternative recognition script
├── .gitignore                     # Git ignore rules
├── requirements.txt              # Python dependencies
└── README.md                     # This file
🛠️ Installation Details
Prerequisites
Python 3.8 or higher

Webcam

Windows/Mac/Linux

Step-by-Step Installation
Clone and setup:

bash
# Clone repository
git clone https://github.com/Arunsingh123481/Hand-gesture-recognition.git
cd Hand-gesture-recognition

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
Verify installation:

bash
python -c "import cv2, mediapipe, sklearn; print('All packages installed successfully!')"
📊 How It Works
Pipeline
text
Webcam Feed → Hand Detection (MediaPipe) → Landmark Extraction → 
Feature Normalization → KNN Classification → Gesture Prediction
Technical Details
Hand Detection: MediaPipe Hands (21 landmarks per hand)

Feature Vector: 63 dimensions (x, y, z coordinates × 21 landmarks)

Classifier: K-Nearest Neighbors (K=5 by default)

Accuracy: >95% with proper training data

Latency: <50ms per frame on average hardware

📖 Usage Guide
1. Real-time Recognition
bash
python gesture_recognition.py
Controls:

Show hand gestures to the webcam

Recognized gesture appears on screen

Press ESC to exit

2. Train Your Own Model
bash
# Step 1: Collect training data
python data_collector.py

# Step 2: Train the model
python train_gesture_model.py

# Step 3: Run recognition
python gesture_recognition.py
3. Data Collection Mode
Run data_collector.py and use these keys:

o → Save OPEN hand gesture

f → Save FIST gesture

p → Save PEACE sign

q → Quit data collection

Tip: Collect 50-100 samples per gesture for best results.

🎭 Available Gestures
Gesture	Key	Image	Description
OPEN	o	✋	All fingers extended
FIST	f	✊	All fingers closed
PEACE	p	✌️	Index and middle fingers extended
Adding More Gestures
To add new gestures (like 👍, 🤘, 👌):

Update data_collector.py:

python
# Add new key bindings
elif key == ord('t'):  # Thumbs up
    csv_writer.writerow(landmarks + ['THUMBS_UP'])
Collect data for the new gesture

Retrain the model:

bash
python train_gesture_model.py
🧠 Model Training
Training Process
The system uses K-Nearest Neighbors algorithm:

python
model = KNeighborsClassifier(
    n_neighbors=5,      # Number of neighbors
    weights='uniform',  # Weight function
    metric='euclidean'  # Distance metric
)
Evaluating Model
After training, check accuracy:

text
Model accuracy: 0.98 (98%)
If accuracy is low:

Collect more training data

Ensure consistent hand positioning

Add more diverse samples

📈 Performance Optimization
For Better Accuracy
Lighting: Ensure good, consistent lighting

Distance: Keep hand 30-50cm from camera

Background: Use plain background initially

Samples: 50-100 samples per gesture minimum

For Faster Performance
Reduce webcam resolution (edit code)

Use fewer neighbors (K=3)

Process every other frame

🔧 Customization
Change Model Parameters
Edit train_gesture_model.py:

python
'''# Try different configurations
model = KNeighborsClassifier(
    n_neighbors=7,           # More neighbors for complex gestures
    weights='distance',      # Weight by distance
    metric='manhattan'       # Different distance metric
)
Add Confidence Threshold
Modify gesture_recognition.py:'''

python
'''# Add confidence check
probabilities = model.predict_proba([landmarks])
confidence = max(probabilities[0])

if confidence > 0.7:  # 70% confidence threshold
    prediction = model.predict([landmarks])[0]
else:
    prediction = "Uncertain" '''

    
🐛 Troubleshooting
Problem	Solution
"No module named 'mediapipe'"	pip install mediapipe
Webcam not detected	Change cv2.VideoCapture(0) to (1)
Poor recognition	Collect more training data
Low FPS	Reduce frame resolution
Model not found	Run train_gesture_model.py first
Common Issues & Fixes
Windows DLL Error:

bash
'''
# Reinstall OpenCV
pip uninstall opencv-python
pip install opencv-python-headless

🚀 Advanced Features
Extend the Project
Gesture-controlled Applications:

Control presentations with gestures

Play games with hand motions

Control smart home devices

Add More Features:

Gesture sequence recognition

Two-hand gesture support

3D gesture tracking

Deploy Options:

Web app with Streamlit

Mobile app with Flutter

Raspberry Pi for embedded systems

🤝 Contributing
We welcome contributions! Here's how:

Fork the repository

Create a feature branch (git checkout -b feature/AmazingFeature)

Commit changes (git commit -m 'Add AmazingFeature')

Push to branch (git push origin feature/AmazingFeature)

Open a Pull Request

Guidelines
Follow PEP 8 style guide

Add comments for complex logic

Update documentation

Test your changes thoroughly
