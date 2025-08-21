import cv2
import mediapipe as mp

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,      # Better for video streams
    max_num_faces=1,              # Detect 1 face (adjust as needed)
    refine_landmarks=True,        # Improved accuracy for eyes/eyebrows/lips [4]
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialize OpenCV webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1200, 800), interpolation=cv2.INTER_CUBIC)
    if not ret:
        break
    
    # Convert BGR to RGB (MediaPipe requires RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process frame with MediaPipe
    results = face_mesh.process(rgb_frame)
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing_styles = mp.solutions.drawing_styles

            # Replace the landmark drawing loop with:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,  # Face mesh
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_tesselation_style()
            )

    
    cv2.imshow('Face Landmarks', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
