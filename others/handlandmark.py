import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,       # Better for video streams
    max_num_hands=2,               # Detect up to 2 hands
    model_complexity=1,            # Higher accuracy (0=light, 1=full, 2=heavy)
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Initialize drawing utilities
mp_drawing = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    frame = cv2.resize(frame, (1200, 800), interpolation=cv2.INTER_CUBIC)
    if not success:
        continue
    
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process frame
    results = hands.process(rgb_frame)
    
    # Draw landmarks if hands detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks and connections
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                mp.solutions.drawing_styles.get_default_hand_connections_style()
            )
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            print(f"Wrist position: X={wrist.x}, Y={wrist.y}, Z={wrist.z}")
            
            # Get thumb tip (Landmark 4)
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

    
    # Display frame
    cv2.imshow('Hand Tracking', cv2.flip(frame, 1))
    
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
