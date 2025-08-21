import cv2
import pygame
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

# Initialize PyGame
pygame.init()
WIDTH, HEIGHT = 1200, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Hand-Controlled Ball")

# Ball properties
BALL_RADIUS = 40
ball_color = (255, 255, 255)  # Red
ball_pos = [WIDTH//2, HEIGHT//2]  # Start at center
SMOOTHING_FACTOR = 0.2
CELL_SIZE = 40
GRIDCOLOR = (40, 40, 40)

# Webcam setup
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

def draw_grid(screen, width=WIDTH, height=HEIGHT, cell_size=CELL_SIZE, color=GRIDCOLOR):
    for x in range(0, width, cell_size):
        pygame.draw.line(screen, color, (x, 0), (x, height))
    for y in range(0, height, cell_size):
        pygame.draw.line(screen, color, (0, y), (width, y))

while True:
    # PyGame event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            cap.release()
            pygame.quit()
            exit()

    # Read webcam frame
    ret, frame = cap.read()
    if not ret:
        continue
    
    # Mirror and process frame
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Hand detection
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get index finger tip (Landmark 8)
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                mp.solutions.drawing_styles.get_default_hand_connections_style()
            )
            print(f"Index position: (X,Y,Z)=[{index_tip.x:.4f},{index_tip.y:.4f},{index_tip.z:.4f}]")
            # Convert normalized coordinates to screen position [1][4]
            ball_pos[0] = int(index_tip.x * WIDTH)
            ball_pos[1] = int(index_tip.y * HEIGHT)
            print(f"Ball Position : (X,Y) = {ball_pos}")
            prev_x, prev_y = ball_pos
            current_x = int(index_tip.x * WIDTH)
            current_y = int(index_tip.y * HEIGHT)

            # Calculate velocity
            velocity_x = (current_x - prev_x) * 2
            velocity_y = (current_y - prev_y) * 2

            # Apply momentum
            ball_pos[0] += velocity_x
            ball_pos[1] += velocity_y

            ball_pos[0] = ball_pos[0] * SMOOTHING_FACTOR + prev_x * (1 - SMOOTHING_FACTOR)

    # PyGame rendering
    screen.fill((0, 0, 0))  # White background
    draw_grid(screen)
    pygame.draw.line(screen, (255, 255, 255), (WIDTH//2, 0), (WIDTH//2, HEIGHT), 2)
    pygame.draw.line(screen, (255, 255, 255), (0, HEIGHT//2), (WIDTH, HEIGHT//2), 2)
    pygame.draw.circle(screen, ball_color, ball_pos, BALL_RADIUS)
    pygame.display.update()

    # Optional: Show webcam feed with landmarks
    cv2.imshow('Hand Tracking', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
