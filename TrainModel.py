import cv2
import mediapipe as mp
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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

# Function to collect data
def collect_data(gesture_label, num_samples=100):
    cap = cv2.VideoCapture(0)
    data = []
    labels = []

    print(f"Collecting data for gesture: {gesture_label}")
    while len(data) < num_samples:
        ret, frame = cap.read()
        if not ret:
            continue

        # Convert the image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Process the image and detect hands
        results = hands.process(image)

        # Draw hand landmarks and collect data
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extract keypoints and add to dataset
                keypoints = extract_keypoints(hand_landmarks)
                data.append(keypoints)
                labels.append(gesture_label)

        # Display the output
        cv2.putText(image, f"Collecting: {gesture_label} ({len(data)}/{num_samples})", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Data Collection', image)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return data, labels

# Function to train the model
def train_model(data, labels):
    
    X = np.array(data)
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    return model

def save_model(model, filename):
    joblib.dump(model, filename)
    print(f"Model saved as {filename}")

def load_model(filename):
    return joblib.load(filename)

def realtime_gesture_recognition(model):
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = hands.process(image)

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
    # Step 1: Collect data for gestures
    gestures = ["one", "two", "three", "four"]
    all_data = []
    all_labels = []

    for gesture in gestures:
        data, labels = collect_data(gesture, num_samples=100)
        all_data.extend(data)
        all_labels.extend(labels)

    model = train_model(all_data, all_labels)

    save_model(model, "gesture_model.pkl")

    loaded_model = load_model("gesture_model.pkl")
    realtime_gesture_recognition(loaded_model)