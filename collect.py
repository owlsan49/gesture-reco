import csv
import os
import time

import cv2
import mediapipe as mp
import numpy as np
import pyrealsense2 as rs

from features import extract_features


MODEL_PATH = "hand_landmarker.task"
DATASET_PATH = "dataset.csv"

LABEL_KEYS = {
    ord("1"): "rock",
    ord("2"): "paper",
    ord("3"): "scissors",
    ord("4"): "thumb_up",
    ord("5"): "ok",
    ord("0"): "none",
}


# ============================
# MediaPipe
# ============================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

# 采集和演示使用相同的配置，避免训练数据和实时推理的检测行为不一致。
MAX_HANDS = 1
MIN_HAND_DETECTION_CONFIDENCE = 0.25
MIN_HAND_PRESENCE_CONFIDENCE = 0.25
MIN_TRACKING_CONFIDENCE = 0.25

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
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

    for point in points:
        cv2.circle(frame, point, 5, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.circle(frame, point, 6, (0, 0, 0), 1, cv2.LINE_AA)

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_hands=MAX_HANDS,
    min_hand_detection_confidence=MIN_HAND_DETECTION_CONFIDENCE,
    min_hand_presence_confidence=MIN_HAND_PRESENCE_CONFIDENCE,
    min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
)

landmarker = HandLandmarker.create_from_options(options)


# ============================
# RealSense
# ============================

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)


current_label = None
recording = False
samples_collected = 0
last_timestamp_ms = 0

print(
    """
Controls:

1 = rock       2 = paper       3 = scissors
4 = thumb_up   5 = ok          0 = none

SPACE = start/pause collection
s     = finish the current label
q     = quit
"""
)


file_exists = os.path.exists(DATASET_PATH)
f = open(DATASET_PATH, "a", newline="")
writer = csv.writer(f)

if not file_exists:
    writer.writerow(["label"] + [f"f{i}" for i in range(63)])


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

        detected_hands = result.hand_landmarks or []
        world_hands = result.hand_world_landmarks or []

        for image_landmarks in detected_hands:
            draw_hand_skeleton(frame, image_landmarks)

        if detected_hands and world_hands:
            features = extract_features(world_hands[0])

            if recording and current_label:
                writer.writerow([current_label] + features.tolist())
                f.flush()
                samples_collected += 1

            cv2.putText(
                frame,
                "HAND FOUND",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
        elif detected_hands:
            # 已经检测到图像中的手，但当前帧没有 3D 点；不写入数据集，
            # 同时继续刷新窗口，方便用户调整距离和姿态。
            cv2.putText(
                frame,
                "HAND DETECTED - WAITING FOR 3D LANDMARKS",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
            )
        else:
            cv2.putText(
                frame,
                "NO HAND",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

        if current_label is None:
            status_text = "NO LABEL SELECTED"
            status_color = (0, 0, 255)
            hint_text = "Press 1-5 / 0 to select gesture"
        elif recording:
            status_text = f"LABEL: {current_label} | RECORDING {samples_collected}"
            status_color = (0, 255, 0)
            hint_text = "Press SPACE to pause, s to finish"
        else:
            status_text = f"LABEL: {current_label} | PAUSED"
            status_color = (0, 255, 255)
            hint_text = "Press SPACE to resume, s to finish"

        cv2.putText(
            frame,
            status_text,
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            status_color,
            2,
        )
        cv2.putText(
            frame,
            hint_text,
            (20, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            status_color,
            2,
        )

        if recording:
            cv2.circle(frame, (600, 30), 12, (0, 0, 255), -1)
            cv2.putText(
                frame,
                "REC",
                (570, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        cv2.imshow("Gesture Dataset Collector", frame)
        key = cv2.waitKey(1) & 0xFF

        if key in LABEL_KEYS:
            current_label = LABEL_KEYS[key]
            samples_collected = 0

            if recording:
                recording = False
                print(f"Switched label to {current_label}; collection paused.")
            else:
                print(f"Selected label {current_label}; press SPACE to start.")

        elif key == ord(" "):
            if current_label is None:
                print("Select a label first (1-5 or 0).")
            else:
                recording = not recording
                if recording:
                    print(f"Recording label: {current_label}")
                else:
                    print(f"Paused; saved {samples_collected} samples.")

        elif key == ord("s"):
            if recording and current_label:
                print(
                    f"Finished {current_label}; saved {samples_collected} samples."
                )
                recording = False
                current_label = None
                samples_collected = 0
            else:
                print("No label is currently being recorded.")

        elif key == ord("q"):
            print("Quitting; saving data and closing resources.")
            break

finally:
    f.close()
    pipeline.stop()
    landmarker.close()
    cv2.destroyAllWindows()
