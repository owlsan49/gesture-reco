import time

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pyrealsense2 as rs

from features import extract_features


MODEL_PATH = "hand_landmarker.task"
CLASSIFIER_PATH = "gesture_svm.joblib"

# 只识别一只手，与采集程序保持一致。
MAX_HANDS = 1

# 默认值是 0.5。降低到 0.25 可以减少暗光、小手和快速移动造成的漏检。
MIN_HAND_DETECTION_CONFIDENCE = 0.25
MIN_HAND_PRESENCE_CONFIDENCE = 0.25
MIN_TRACKING_CONFIDENCE = 0.25

# 这是“手势分类”的拒识阈值，不是“有没有手”的检测阈值。
GESTURE_CONFIDENCE_THRESHOLD = 0.75

# MediaPipe Hand Landmarker 的 21 个关键点连接关系。
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),       # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # index
    (5, 9), (9, 10), (10, 11), (11, 12),  # middle
    (9, 13), (13, 14), (14, 15), (15, 16),  # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),
)


def draw_hand_skeleton(frame, landmarks):
    """将归一化图像坐标中的 21 个关键点绘制到 BGR 帧上。"""

    height, width = frame.shape[:2]
    points = []

    for landmark in landmarks:
        x = int(np.clip(landmark.x * width, 0, width - 1))
        y = int(np.clip(landmark.y * height, 0, height - 1))
        points.append((x, y))

    for start, end in HAND_CONNECTIONS:
        cv2.line(
            frame,
            points[start],
            points[end],
            (255, 180, 0),
            2,
            cv2.LINE_AA,
        )

    for index, point in enumerate(points):
        cv2.circle(frame, point, 5, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.circle(frame, point, 6, (0, 0, 0), 1, cv2.LINE_AA)


classifier = joblib.load(CLASSIFIER_PATH)


# =========================
# MediaPipe
# =========================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_hands=MAX_HANDS,
    min_hand_detection_confidence=MIN_HAND_DETECTION_CONFIDENCE,
    min_hand_presence_confidence=MIN_HAND_PRESENCE_CONFIDENCE,
    min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
)

landmarker = HandLandmarker.create_from_options(options)


# =========================
# RealSense D455
# =========================

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

last_timestamp_ms = 0

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # VIDEO 模式要求时间戳严格递增。
        timestamp_ms = int(time.monotonic() * 1000)
        timestamp_ms = max(timestamp_ms, last_timestamp_ms + 1)
        last_timestamp_ms = timestamp_ms

        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        # image landmarks 表示“检测到了手”；world landmarks 只用于生成
        # 与训练集一致的 63 维特征。不要用后者单独判断是否有手。
        detected_hands = result.hand_landmarks or []
        world_hands = result.hand_world_landmarks or []
        display_lines = ["No hand"]

        if detected_hands:
            display_lines = []

            for hand_index, image_landmarks in enumerate(detected_hands):
                draw_hand_skeleton(frame, image_landmarks)

                if hand_index >= len(world_hands):
                    display_lines.append(
                        f"Hand {hand_index + 1} detected (no 3D landmarks)"
                    )
                    continue

                landmarks = world_hands[hand_index]

                features = extract_features(landmarks)
                x = features.reshape(1, -1)

                probs = classifier.predict_proba(x)[0]
                class_index = int(np.argmax(probs))
                label = classifier.classes_[class_index]
                gesture_confidence = float(probs[class_index])
                confidence_text = f"svm {gesture_confidence:.2f}"

                if gesture_confidence < GESTURE_CONFIDENCE_THRESHOLD:
                    label = "none"

                side = "Unknown"
                hand_confidence = 0.0
                if hand_index < len(result.handedness):
                    categories = result.handedness[hand_index]
                    if categories:
                        side = categories[0].category_name
                        hand_confidence = float(categories[0].score)

                display_lines.append(
                    f"Hand {hand_index + 1} [{side} {hand_confidence:.2f}] "
                    f"{label}: {confidence_text}"
                )

            print(" | ".join(display_lines))

        for line_index, text in enumerate(display_lines):
            color = (0, 255, 0) if detected_hands else (0, 0, 255)
            cv2.putText(
                frame,
                text,
                (20, 50 + line_index * 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                color,
                2,
            )

        cv2.imshow("Custom Gesture", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    pipeline.stop()
    landmarker.close()
    cv2.destroyAllWindows()
