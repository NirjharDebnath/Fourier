import cv2
import mediapipe as mp

# Initialize MediaPipe Hands and Gesture Recognizer [4][7]
mp_hands = mp.solutions.hands
mp_gesture = mp.tasks.vision.GestureRecognizer

# Download model from https://storage.googleapis.com/mediapipe-assets/gesture_recognizer.task
model_path = 'gesture_recognizer.task'

# Configuration [4]
BaseOptions = mp.tasks.BaseOptions
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO
)

# Initialize recognizer
with mp_gesture.create_from_options(options) as recognizer:
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.resize(frame, (1600, 1000), interpolation=cv2.INTER_CUBIC)
        if not ret: break
        
        # Convert to MediaPipe Image format
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Process frame at 30 FPS (timestamp in ms)
        result = recognizer.recognize_for_video(mp_image, int(cap.get(cv2.CAP_PROP_POS_MSEC)))
        
        # Draw results
        if result.gestures:
            for gesture, hand in zip(result.gestures, result.hand_landmarks):
                # Get gesture label and confidence [4][7]
                label = gesture[0].category_name
                confidence = gesture[0].score
                
                # Draw landmarks
                for landmark in hand:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x,y), 5, (0,255,0), -1)
                
                # Display gesture
                cv2.putText(frame, f"{label} ({confidence:.2f})", (10,30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        
        cv2.imshow('Gesture Control', frame)
        if cv2.waitKey(1) == ord('q'): break

cap.release()
cv2.destroyAllWindows()
