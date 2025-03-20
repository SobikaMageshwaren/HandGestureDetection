Real-Time Hand and Pose Detection with Streamlit
This project uses OpenCV, Mediapipe, and Streamlit to build a web-based real-time hand gesture and pose detection application.

🎯 Features
Detects hand gestures (Thumbs Up, OK Sign, Open Hand, Fist)
Detects elbow joints from the pose landmarks
Visualizes the landmarks directly in the Streamlit interface
Supports real-time webcam video feed

🛠️ Setup and Installation

1️⃣ Install Dependencies
Ensure you have Python 3.8 or newer. Then install the required packages:

2️⃣ Save the Code
Create a Python file named app.py and paste the full code from the script above.

3️⃣ Run the Streamlit Application
Launch your app with this command: python -m streamlit app.py
This will open your default browser with the Streamlit interface.

🔧 How to Use
Click the "Start Detection" button.
Allow webcam permissions.
The app will display:
Hand landmarks and recognized gestures (e.g., Thumbs Up, OK Sign)
Pose landmarks, specifically marking elbows
To quit, either:
Close the Streamlit tab
Press 'q' on your keyboard

🤖 Gesture Recognition Rules

Gesture
Condition
Thumbs Up
Thumb close to index finger, wrist higher than pinky
OK Sign
Thumb close to index finger, wrist lower than pinky
Open Hand
Wrist far from thumb (open palm)
Fist
Wrist close to thumb


🧠 Troubleshooting

Webcam not detected: Ensure no other apps (e.g., Zoom) are using the camera.
Streamlit fails to load: Ensure the packages are installed correctly. Try: pip show streamlit
Gesture not recognized well: Adjust the gesture thresholds inside detect_gesture().

📌 Acknowledgments

Mediapipe by Google for fast, robust hand/pose detection
OpenCV for real-time computer vision
Streamlit for making interactive web apps easy
