import cv2
import mediapipe as mp
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import math
import threading
import time

# ==========================================
# SETTINGS AND STARTING VALUES
# ==========================================
SCREEN_WIDTH, SCREEN_HEIGHT = 1000, 700
TOTAL_DOTS = 1500  # how many dots (particles) we draw

# Data shared between the camera thread and the drawing (main) thread
data_lock = threading.Lock()
shared_info = {
    "mode": 1,
    "hand_target_x": 0.0,
    "hand_target_y": 0.0,
    "hand_target_z": -12.0,
    "camera_frame": None,
    "is_running": True
}

# ==========================================
# MAKING THE 3D SHAPES (DOTS)
# ==========================================

# Mode 1: Space (dots scattered randomly)
space_shape = np.random.uniform(-4.0, 4.0, (TOTAL_DOTS, 3))

# Mode 2: Saturn (a ball in the middle + a ring around it)
saturn_shape = np.zeros((TOTAL_DOTS, 3))
BALL_DOTS = 700  # dots used for the ball, the rest make the ring
for i in range(BALL_DOTS):
    angle_around = np.random.uniform(0, 2 * np.pi)
    cos_of_angle_down = np.random.uniform(-1, 1)
    angle_down = np.arccos(cos_of_angle_down)
    ball_radius = 1.3
    saturn_shape[i, 0] = ball_radius * np.sin(angle_down) * np.cos(angle_around)
    saturn_shape[i, 1] = ball_radius * np.sin(angle_down) * np.sin(angle_around)
    saturn_shape[i, 2] = ball_radius * np.cos(angle_down)

for i in range(BALL_DOTS, TOTAL_DOTS):
    ring_angle = np.random.uniform(0, 2 * np.pi)
    ring_radius = np.random.uniform(1.8, 3.8)
    saturn_shape[i, 0] = ring_radius * np.cos(ring_angle)
    saturn_shape[i, 1] = np.random.uniform(-0.05, 0.05)  # very thin ring
    saturn_shape[i, 2] = ring_radius * np.sin(ring_angle)

def make_text_shape(first_line, second_line):
    """Writes two lines of text and turns the letters into dots (so it fits the screen)."""
    text_picture = np.zeros((350, 800), dtype=np.uint8)
    cv2.putText(text_picture, first_line, (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 3.0, 255, 9, cv2.LINE_AA)
    cv2.putText(text_picture, second_line, (20, 300), cv2.FONT_HERSHEY_SIMPLEX, 3.0, 255, 9, cv2.LINE_AA)
    pixel_rows, pixel_columns = np.where(text_picture > 0)

    # Move the text to the center of the screen
    text_x = (pixel_columns - (pixel_columns.min() + pixel_columns.max()) / 2) / 75.0
    text_y = -(pixel_rows - (pixel_rows.min() + pixel_rows.max()) / 2) / 75.0
    text_z = np.random.uniform(-0.1, 0.1, len(text_x))
    all_text_points = np.stack((text_x, text_y, text_z), axis=-1)
    picked_numbers = np.random.choice(len(all_text_points), TOTAL_DOTS)
    return all_text_points[picked_numbers]


# Mode 3: Text "MEN ARE / BRAVE"
text_shape = make_text_shape("MEN ARE", "BRAVE")

# Mode 5: Text "I AM / BATMAN"
batman_text_shape = make_text_shape("I AM", "BATMAN")

# Mode 6: Text "GOTHAM / IS MINE"
gotham_text_shape = make_text_shape("GOTHAM", "IS MINE")

# Mode 7: Text "I AM / VENGEANCE"
vengeance_text_shape = make_text_shape("I AM", "VENGEANCE")


# Mode 4: Batman logo
def make_batman_shape(total_dots):
    """Fills the Batman logo (a known math curve) with random dots."""
    # Throw many random points in a box, then keep only the ones inside the logo
    point_x = np.random.uniform(-7.0, 7.0, 200000)
    point_y = np.random.uniform(-3.5, 3.3, 200000)
    abs_x = np.abs(point_x)

    # Top edge of the logo (ears, shoulders and wings)
    top_edge = np.where(
        abs_x >= 3,
        3 * np.sqrt(np.clip(1 - (point_x / 7) ** 2, 0, None)),
        np.where(
            abs_x >= 1,
            6 * np.sqrt(10) / 7 + (3 - abs_x) / 2
            - (3 * np.sqrt(10) / 14) * np.sqrt(np.clip(4 - (abs_x - 1) ** 2, 0, None)),
            np.where(
                abs_x >= 0.75,
                9 - 8 * abs_x,
                np.where(abs_x >= 0.5, 3 * abs_x + 0.75, 2.25)
            )
        )
    )

    # Bottom edge of the logo (wing tips and the pointed middle)
    bottom_edge = np.where(
        abs_x > 4,
        -3 * np.sqrt(np.clip(1 - (point_x / 7) ** 2, 0, None)),
        abs_x / 2
        - ((3 * np.sqrt(33) - 7) / 112) * point_x ** 2
        - 3
        + np.sqrt(np.clip(1 - (np.abs(abs_x - 2) - 1) ** 2, 0, None))
    )

    is_inside = (point_y <= top_edge) & (point_y >= bottom_edge)
    inside_x = point_x[is_inside]
    inside_y = point_y[is_inside]

    chosen = np.random.choice(len(inside_x), total_dots, replace=False)
    logo_scale = 0.38  # makes the logo smaller to fit the screen
    batman_points = np.zeros((total_dots, 3))
    batman_points[:, 0] = inside_x[chosen] * logo_scale
    batman_points[:, 1] = inside_y[chosen] * logo_scale
    batman_points[:, 2] = np.random.uniform(-0.2, 0.2, total_dots)  # a little thickness
    return batman_points


batman_shape = make_batman_shape(TOTAL_DOTS)

dots_now = np.copy(space_shape)      # where the dots are right now
dots_goal = np.copy(space_shape)     # where the dots want to go


# ==========================================
# FUNCTION TO UNDERSTAND THE HAND GESTURE
# ==========================================
def thumb_is_out(hand_points):
    """True if the thumb is pointing away from the palm, False if it is folded in."""
    def distance_between(point_a, point_b):
        return math.sqrt((point_a.x - point_b.x) ** 2 + (point_a.y - point_b.y) ** 2)

    thumb_tip = hand_points.landmark[4]
    pinky_base = hand_points.landmark[17]
    index_base = hand_points.landmark[5]
    palm_width = distance_between(index_base, pinky_base)
    # Thumb far from the pinky side of the palm = thumb is out
    return distance_between(thumb_tip, pinky_base) > palm_width * 1.2


def find_hand_position(hand_points):
    """Returns the x, y and z place on screen where the 3D shape should be drawn for this hand."""
    wrist = hand_points.landmark[0]
    hand_x = (wrist.x - 0.5) * 10.0
    hand_y = -(wrist.y - 0.5) * 7.0

    # The bigger the hand looks, the closer it is to the camera
    pinky_base = hand_points.landmark[17]
    hand_size = math.sqrt((wrist.x - pinky_base.x) ** 2 + (wrist.y - pinky_base.y) ** 2)
    hand_z = -10.0 - (1.0 / (hand_size + 0.01)) * 0.2
    return hand_x, hand_y, hand_z


def find_mode_from_gesture(hand_points):
    finger_tip_numbers = [8, 12, 16, 20]   # index, middle, ring, pinky tips
    finger_middle_numbers = [6, 10, 14, 18]  # the middle joint of each finger
    # A finger is "up" when its tip is higher on screen than its middle joint
    fingers_up = [
        hand_points.landmark[tip].y < hand_points.landmark[middle].y
        for tip, middle in zip(finger_tip_numbers, finger_middle_numbers)
    ]

    if sum(fingers_up) == 0:
        return 4  # Closed fist -> Batman logo
    if fingers_up[0] and not any(fingers_up[1:]):
        return 2  # One finger -> Saturn
    if fingers_up[0] and fingers_up[1] and not fingers_up[2] and not fingers_up[3]:
        return 3  # Peace sign -> Text "MEN ARE BRAVE"
    if fingers_up[0] and fingers_up[1] and fingers_up[2] and not fingers_up[3]:
        return 5  # Three fingers (thumb and little finger down) -> Text "I AM BATMAN"
    if all(fingers_up) and not thumb_is_out(hand_points):
        return 6  # Four fingers up, thumb folded in -> Text "GOTHAM IS MINE"
    return 1      # Anything else (open hand) -> Space


# ==========================================
# BACKGROUND THREAD: CAMERA AND HAND DETECTION
# ==========================================
def run_camera_loop():
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # Small wait so the camera (and its permission prompt) can start
    cv2.waitKey(500)

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    mp_hands = mp.solutions.hands
    hand_finder = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    while shared_info["is_running"]:
        got_frame, frame = camera.read()
        if not got_frame:
            time.sleep(0.01)
            continue

        frame = cv2.flip(frame, 1)  # mirror the picture
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hand_finder.process(rgb_frame)

        new_mode = 1
        new_x, new_y, new_z = 0.0, 0.0, -12.0

        if results.multi_hand_landmarks:
            all_modes = []
            all_positions = []
            for hand_points in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_points, mp_hands.HAND_CONNECTIONS)
                all_modes.append(find_mode_from_gesture(hand_points))
                all_positions.append(find_hand_position(hand_points))

            if len(all_modes) == 2 and all_modes[0] == 4 and all_modes[1] == 4:
                # Both hands are fists -> "I AM VENGEANCE", placed between the two hands
                new_mode = 7
                new_x = (all_positions[0][0] + all_positions[1][0]) / 2
                new_y = (all_positions[0][1] + all_positions[1][1]) / 2
                new_z = (all_positions[0][2] + all_positions[1][2]) / 2
            else:
                # Otherwise only the first detected hand controls the app
                new_mode = all_modes[0]
                new_x, new_y, new_z = all_positions[0]

        with data_lock:
            shared_info["mode"] = new_mode
            shared_info["hand_target_x"] = new_x
            shared_info["hand_target_y"] = new_y
            shared_info["hand_target_z"] = new_z
            shared_info["camera_frame"] = frame

    camera.release()


# Start the camera thread first
camera_thread = threading.Thread(target=run_camera_loop, daemon=True)
camera_thread.start()

# Wait 1 second so the camera is ready before the window opens
time.sleep(1.0)

# ==========================================
# MAIN THREAD: DRAWING THE 3D GRAPHICS
# ==========================================
pygame.init()
pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), DOUBLEBUF | OPENGL)
pygame.display.set_caption("Space Gesture Controller")

glMatrixMode(GL_PROJECTION)
glLoadIdentity()
gluPerspective(45, (SCREEN_WIDTH / SCREEN_HEIGHT), 0.1, 50.0)
glMatrixMode(GL_MODELVIEW)
glEnable(GL_DEPTH_TEST)

game_clock = pygame.time.Clock()
spin_angle = 0.0
swing_timer = 0.0  # used to gently swing the Batman logo left and right
hand_x, hand_y, hand_z = 0.0, 0.0, -12.0

mode_names = {
    1: "SPACE SPIN (Open hand)",
    2: "SATURN 3D (One finger)",
    3: "TEXT MEN ARE BRAVE (Peace sign)",
    4: "BATMAN LOGO (Fist)",
    5: "TEXT I AM BATMAN (Three fingers)",
    6: "TEXT GOTHAM IS MINE (Four fingers, thumb in)",
    7: "TEXT I AM VENGEANCE (Two fists)"
}

while shared_info["is_running"]:
    pygame.event.pump()
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
            shared_info["is_running"] = False

    with data_lock:
        current_mode = shared_info["mode"]
        hand_target_x = shared_info["hand_target_x"]
        hand_target_y = shared_info["hand_target_y"]
        hand_target_z = shared_info["hand_target_z"]
        camera_frame = shared_info["camera_frame"]

    if camera_frame is not None:
        cv2.putText(camera_frame, f"MODE: {mode_names[current_mode]}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow("Hand Sensor Monitor", camera_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            shared_info["is_running"] = False

    glClearColor(0.0, 0.0, 0.0, 1.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()

    # Move the shape smoothly toward the hand position
    hand_x += (hand_target_x - hand_x) * 0.25
    hand_y += (hand_target_y - hand_y) * 0.25
    hand_z += (hand_target_z - hand_z) * 0.25

    if current_mode == 1:
        dots_goal = space_shape
        spin_angle += 0.5
    elif current_mode == 2:
        dots_goal = saturn_shape
        spin_angle += 2.0
    elif current_mode == 3:
        dots_goal = text_shape
        spin_angle = 0.0  # text stays still so it can be read
    elif current_mode == 4:
        dots_goal = batman_shape
        swing_timer += 0.03
        spin_angle = math.sin(swing_timer) * 30.0  # swing +-30 degrees so the logo stays readable
    elif current_mode == 5:
        dots_goal = batman_text_shape
        spin_angle = 0.0  # text stays still so it can be read
    elif current_mode == 6:
        dots_goal = gotham_text_shape
        spin_angle = 0.0  # text stays still so it can be read
    elif current_mode == 7:
        dots_goal = vengeance_text_shape
        spin_angle = 0.0  # text stays still so it can be read

    # Dots slowly fly from their old place to the new shape
    dots_now += (dots_goal - dots_now) * 0.15

    if current_mode in [2, 3, 4, 5, 6, 7]:
        glTranslatef(hand_x, hand_y, hand_z)
        if current_mode == 2:
            glRotatef(25, 1.0, 0.0, 0.5)
    else:
        glTranslatef(0.0, 0.0, -12.0)

    glRotatef(spin_angle, 0.0, 1.0, 0.0)

    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glPointSize(4.5)

    glBegin(GL_POINTS)
    for i in range(TOTAL_DOTS):
        if current_mode == 3:
            glColor4f(0.2, 0.4, 1.0, 0.95)     # "MEN ARE BRAVE" text: classic Batman blue
        elif current_mode == 4:
            glColor4f(1.0, 0.85, 0.0, 0.95)    # Batman logo: yellow
        elif current_mode == 5:
            glColor4f(1.0, 0.85, 0.0, 0.95)    # "I AM BATMAN" text: yellow
        elif current_mode == 6:
            glColor4f(0.7, 0.7, 0.75, 0.95)    # "GOTHAM IS MINE" text: steel grey
        elif current_mode == 7:
            glColor4f(1.0, 0.1, 0.1, 0.95)     # "I AM VENGEANCE" text: red
        elif current_mode == 2 and i >= BALL_DOTS:
            glColor4f(0.4, 0.9, 0.85, 0.6)     # Saturn ring: light teal
        elif current_mode == 2 and i < BALL_DOTS:
            glColor4f(0.0, 0.8, 0.7, 0.85)     # Saturn ball: teal
        else:
            glColor4f(0.7, 0.7, 0.75, 0.8)     # space: steel grey

        glVertex3f(dots_now[i, 0], dots_now[i, 1], dots_now[i, 2])
    glEnd()

    pygame.display.flip()
    game_clock.tick(60)

cv2.destroyAllWindows()
pygame.quit()