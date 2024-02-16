import csv
import copy
import itertools
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

# ランドマークの画像上の位置を算出する関数
def calc_landmark_list(image, landmarks):
    landmark_point = []
    image_width, image_height = image.shape[1], image.shape[0]

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

# 座標履歴を描画する関数
def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv2.circle(image, (point[0], point[1]), 1 + int(index / 2), (255, 0, 0), 2)
    return image

def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    # 相対座標に変換
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (temp_point_history[index][0] - base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] - base_y) / image_height

    # 1次元リストに変換
    temp_point_history = list(itertools.chain.from_iterable(temp_point_history))

    del temp_point_history[0:2]

    return temp_point_history

# カメラキャプチャ設定
camera_no = 0
video_capture = cv2.VideoCapture(camera_no)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# MediaPipe Pose初期化
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# トラック対象の設定 (肩と手)
ID_SHOULDER = 11  # 右肩
ID_HAND = 20  # 右手首

# 座標履歴を保持するための変数
history_length = 16
point_history = deque(maxlen=history_length)

# ジェスチャー認識用モデルロード
tflite_save_path = './data/gesture_classifier.tflite'
interpreter = tf.lite.Interpreter(model_path=tflite_save_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

gesture_label = ['0', '1', '2']

while video_capture.isOpened():
    # カメラ画像取得
    ret, frame = video_capture.read()
    if ret is False:
        break
    frame_width, frame_height = frame.shape[1], frame.shape[0]

    # 鏡映しになるよう反転
    frame = cv2.flip(frame, 1)

    # MediaPipeで扱う画像は、OpenCVのBGRの並びではなくRGBのため変換
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 画像をリードオンリーにしてPose検出処理実施
    rgb_image.flags.writeable = False
    pose_results = pose.process(rgb_image)
    rgb_image.flags.writeable = True

    # 有効なポーズが検出された場合、ポーズを描画
    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # ポーズの座標計算
        shoulder_pos = calc_landmark_list(rgb_image, pose_results.pose_landmarks)[ID_SHOULDER]
        hand_pos = calc_landmark_list(rgb_image, pose_results.pose_landmarks)[ID_HAND]

        # 肩と手の位置座標を履歴に追加
        point_history.append(shoulder_pos)
        point_history.append(hand_pos)

    if len(point_history) == history_length:
        temp_point_history = pre_process_point_history(rgb_image, point_history)

        interpreter.set_tensor(
            input_details[0]['index'],
            np.array([temp_point_history]).astype(np.float32))
        interpreter.invoke()
        tflite_results = interpreter.get_tensor(output_details[0]['index'])

        gesture_id = np.argmax(np.squeeze(tflite_results))

        # ディスプレイ表示
        cv2.putText(frame, gesture_label[gesture_id], (10, 30),
                    cv2.FONT_HERSHEY_PLAIN, 10, (100, 100, 100), 1, cv2.LINE_AA)
        frame = draw_point_history(frame, point_history)
        cv2.imshow('full_body_detection', frame)

    # キー入力(ESC:プログラム終了)
    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break

# リソースの解放
video_capture.release()
pose.close()
cv2.destroyAllWindows()
