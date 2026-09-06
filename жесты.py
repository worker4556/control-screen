import collections
import math
import os
import subprocess
import time
import cv2
import mediapipe as mp
import pyautogui
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python.solutions import drawing_utils
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import HandLandmarksConnections

# ==========================================
# 1. КЛАСС ДЕТЕКЦИИ СВАЙПОВ
# ==========================================
class SwipeDetector:

    def __init__(
        self, min_distance=0.18, max_duration=0.40, dominant_axis_ratio=2.0
    ):
        self.min_distance = min_distance
        self.max_duration = max_duration
        self.axis_ratio = dominant_axis_ratio
        self.history = collections.deque(maxlen=10)
        self.last_swipe_time = 0
        self.cooldown = 0.5

    def update(self, x, y):
        now = time.time()
        self.history.append((x, y, now))

        if now - self.last_swipe_time < self.cooldown:
            return None

        if len(self.history) < 3:
            return None

        start_x, start_y, start_time = self.history[0]
        curr_x, curr_y, curr_time = self.history[-1]

        dt = curr_time - start_time
        if dt > self.max_duration or dt == 0:
            return None

        dx = curr_x - start_x
        dy = curr_y - start_y

        if abs(dx) < self.min_distance:
            return None

        if abs(dx) < abs(dy) * self.axis_ratio:
            return None

        self.last_swipe_time = now
        self.history.clear()

        # По координатам медиапайпа: dx > 0 это движение вправо кадра
        return "RIGHT" if dx > 0 else "LEFT"


# ==========================================
# 2. КОНФИГУРАЦИЯ И КОНСТАНТЫ
# ==========================================
MODEL_PATH = "gesture_recognizer.task"
CAMERA_INDEX = 0

GESTURE_FIST = "Closed_Fist"
GESTURE_VICTORY = "Victory"
GESTURE_THUMB_UP = "Thumb_Up"
GESTURE_THUMB_DOWN = "Thumb_Down"

GESTURE_COOLDOWN = 1.2
last_gesture_time = {}

HAND_CONNECTIONS = [
    (c.start, c.end) for c in HandLandmarksConnections.HAND_CONNECTIONS
]

swipe_detector = SwipeDetector(min_distance=0.18, max_duration=0.40)


# ==========================================
# 3. СИСТЕМНЫЕ ФУНКЦИИ (macOS)
# ==========================================
def can_execute(gesture):
    now = time.time()
    if gesture not in last_gesture_time:
        return True
    return (now - last_gesture_time[gesture]) >= GESTURE_COOLDOWN


def toggle_audio_mute():
    try:
        script = """
        if output muted of (get volume settings) is true then
            set volume without output muted
        else
            set volume with output muted
        end if
        """
        subprocess.run(["osascript", "-e", script], check=True)
        print("✓ Звук переключен (Mute/Unmute)")
        return True
    except Exception as e:
        print(f"Ошибка переключения звука: {e}")
        return False


def scroll_down():
    pyautogui.scroll(-15)
    print("↓ Скролл вниз")


def scroll_up():
    pyautogui.scroll(15)
    print("↑ Скролл вверх")


def launch_firefox():
    try:
        subprocess.Popen(["open", "-a", "Firefox"])
        print("🌍 Firefox запущен")
        return True
    except Exception as e:
        print(f"Ошибка запуска Firefox: {e}")
        return False


def switch_desktop(direction):
    """Переключение рабочих столов в macOS."""
    if direction == "RIGHT":
        script = (
            'tell application "System Events" to key code 124 using control down'
        )
        print("➡️ Свайп ВПРАВО: Следующий рабочий стол")
    elif direction == "LEFT":
        script = (
            'tell application "System Events" to key code 123 using control down'
        )
        print("⬅️ Свайп ВЛЕВО: Предыдущий рабочий стол")

    try:
        subprocess.run(["osascript", "-e", script], check=True)
    except Exception as e:
        print(f"Ошибка выполнения свайпа: {e}")


def execute_gesture(gesture):
    if not can_execute(gesture):
        return

    if gesture == GESTURE_FIST:
        if toggle_audio_mute():
            last_gesture_time[GESTURE_FIST] = time.time()
    elif gesture == GESTURE_VICTORY:
        if launch_firefox():
            last_gesture_time[GESTURE_VICTORY] = time.time()
    elif gesture == GESTURE_THUMB_DOWN:
        scroll_down()
        last_gesture_time[GESTURE_THUMB_DOWN] = time.time()
    elif gesture == GESTURE_THUMB_UP:
        scroll_up()
        last_gesture_time[GESTURE_THUMB_UP] = time.time()


# ==========================================
# 4. МОДУЛЬ АНАЛИЗА ПОЗЫ И ТРЕВОГИ
# ==========================================
class BodyAnalyzer:

    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        self.prev_shoulder_pos = None
        self.prev_time = time.time()
        self.last_alarm_time = 0

    def process(self, rgb_frame):
        return self.pose.process(rgb_frame).pose_landmarks

    def check_threats(self, landmarks):
        if not landmarks:
            return None

        now = time.time()
        dt = max(now - self.prev_time, 0.001)

        left_s = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_s = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        curr_x = (left_s.x + right_s.x) / 2
        curr_y = (left_s.y + right_s.y) / 2

        is_shaking = False
        if self.prev_shoulder_pos:
            dist = math.sqrt(
                (curr_x - self.prev_shoulder_pos[0]) ** 2
                + (curr_y - self.prev_shoulder_pos[1]) ** 2
            )
            velocity = dist / dt
            if velocity > 1.8:
                is_shaking = True

        self.prev_shoulder_pos = (curr_x, curr_y)
        self.prev_time = now

        nose = landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
        left_w = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
        right_w = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]

        hands_up = (left_w.y < nose.y) and (right_w.y < nose.y)

        if is_shaking:
            return "ОБНАРУЖЕНА СИЛЬНАЯ ТРЯСКА!"
        elif hands_up:
            return "ТРЕВОГА: РУКИ ВВЕРХ (SOS)!"

        return None

    def trigger_alarm(self, reason):
        now = time.time()
        if now - self.last_alarm_time > 2.0:
            print(f"🚨 [ALARM]: {reason}")
            self.last_alarm_time = now


# ==========================================
# 5. ИНИЦИАЛИЗАЦИЯ И ГЛАВНЫЙ ЦИКЛ
# ==========================================
base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = mp_vision.GestureRecognizerOptions(
    base_options=base_options,
    running_mode=mp_vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
gesture_recognizer = mp_vision.GestureRecognizer.create_from_options(options)

body_analyzer = BodyAnalyzer()
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("Ошибка открытия камеры.")
    exit()

frame_timestamp_ms = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    frame_timestamp_ms += 33

    # --- ЭТАП 1: Анализ тела (Pose) ---
    pose_landmarks = body_analyzer.process(rgb_frame)
    if pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, pose_landmarks, mp_pose.POSE_CONNECTIONS
        )
        threat_msg = body_analyzer.check_threats(pose_landmarks)
        if threat_msg:
            body_analyzer.trigger_alarm(threat_msg)
            cv2.putText(
                frame,
                threat_msg,
                (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

    # --- ЭТАП 2: Анализ жестов рук (Hands/Gestures) ---
    result = gesture_recognizer.recognize_for_video(
        mp_image, frame_timestamp_ms
    )

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            proto = landmark_pb2.NormalizedLandmarkList()
            proto.landmark.extend(
                [
                    landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)
                    for lm in hand_landmarks
                ]
            )
            drawing_utils.draw_landmarks(frame, proto, HAND_CONNECTIONS)

        # ДЕТЕКЦИЯ СВАЙПОВ (внутри блока проверки наличия рук)
        hand = result.hand_landmarks[0]
        index_tip = hand[8]  # Кончик указательного пальца

        raw_swipe = swipe_detector.update(index_tip.x, index_tip.y)

        if raw_swipe:
            # Инвертируем, так как кадр отзеркален через cv2.flip
            actual_swipe = "RIGHT" if raw_swipe == "LEFT" else "LEFT"

            switch_desktop(actual_swipe)
            cv2.putText(
                frame,
                f"SWIPE {actual_swipe}!",
                (50, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (255, 0, 255),
                3,
            )

    # Распознавание статических жестов
    if (
        result.gestures
        and len(result.gestures) > 0
        and len(result.gestures[0]) > 0
    ):
        top_gesture = result.gestures[0][0]
        gesture_name = top_gesture.category_name

        execute_gesture(gesture_name)

        label = f"{gesture_name} ({top_gesture.score:.0%})"
        cv2.putText(
            frame,
            label,
            (10, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    cv2.imshow("Security & Gesture Control", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()