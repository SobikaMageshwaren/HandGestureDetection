import cv2
import mediapipe as mp
import streamlit as st

# Initialize Mediapipe Hand and Pose Detection
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

# Streamlit Dashboard setup
st.title("Real-Time Hand and Joint Detection")
run = st.button("Start Detection")

# Set up the webcam feed
cap = cv2.VideoCapture(0)

# Display area in Streamlit
frame_placeholder = st.empty()
detection_placeholder = st.empty()

# Function to detect hand gestures
def detect_gesture(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    middle_tip = hand_landmarks.landmark[12]
    ring_tip = hand_landmarks.landmark[16]
    pinky_tip = hand_landmarks.landmark[20]
    wrist = hand_landmarks.landmark[0]

    # Calculate distances
    thumb_index_dist = abs(thumb_tip.x - index_tip.x)
    wrist_thumb_dist = abs(wrist.y - thumb_tip.y)
    wrist_pinky_dist = abs(wrist.y - pinky_tip.y)

    # Recognize gestures
    if thumb_index_dist < 0.05 and wrist_thumb_dist < wrist_pinky_dist:
        return "Thumbs Up"
    if thumb_index_dist < 0.05 and wrist_thumb_dist > wrist_pinky_dist:
        return "OK Sign"
    return "Open Hand" if wrist_thumb_dist > 0.1 else "Fist"

if run:
    with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
        with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to open camera.")
                    break

                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Process hand detection
                hand_results = hands.process(rgb_frame)
                pose_results = pose.process(rgb_frame)

                detection_text = []

                # Draw hand landmarks if detected
                if hand_results.multi_hand_landmarks:
                    for hand_landmarks in hand_results.multi_hand_landmarks:
                        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                        # Gesture recognition
                        gesture = detect_gesture(hand_landmarks)
                        detection_text.append(f"Gesture: {gesture}")
                        h, w, _ = frame.shape
                        cv2.putText(frame, f"Gesture: {gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Draw pose landmarks if detected
                if pose_results.pose_landmarks:
                    mp_draw.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    for idx, landmark in enumerate(pose_results.pose_landmarks.landmark):
                        if idx in [7, 8]:  # Elbows (7 = left elbow, 8 = right elbow)
                            h, w, _ = frame.shape
                            cx, cy = int(landmark.x * w), int(landmark.y * h)
                            cv2.putText(frame, "Elbow", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                            detection_text.append(f"Elbow detected at ({cx}, {cy})")

                # Display the output in Streamlit
                frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
                detection_placeholder.write("\n".join(detection_text))

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

# Release resources
cap.release()
cv2.destroyAllWindows()
