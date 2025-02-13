import cv2
import mediapipe as mp
import numpy as np
import joblib

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Function to extract hand keypoints
def extract_keypoints(hand_landmarks):
    keypoints = []
    for landmark in hand_landmarks.landmark:
        keypoints.extend([landmark.x, landmark.y, landmark.z])  # Use x, y, z coordinates
    return keypoints

# Load the trained model
def load_model(filename):
    return joblib.load(filename)

# Function for real-time gesture recognition
def realtime_gesture_recognition(model):
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        # Convert the image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Process the image and detect hands
        results = hands.process(image)

        # Draw hand landmarks and recognize gestures
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extract keypoints and predict gesture
                keypoints = extract_keypoints(hand_landmarks)
                gesture = model.predict([keypoints])[0]
                cv2.putText(image, f"Gesture: {gesture}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the output
        cv2.imshow('Gesture Recognition', image)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Main program
if __name__ == "__main__":
    # Load the trained model
    model = load_model("gesture_model.pkl")

    # Run real-time gesture recognition
    realtime_gesture_recognition(model)