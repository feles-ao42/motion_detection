# -*- coding: utf-8 -*-
from collections import deque

import cv2
import mediapipe as mp


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
            cv2.circle(image, (point[0], point[1]), 1 + int(index / 2),
                       (255, 0, 0), 2)
    return image


# カメラキャプチャ設定
camera_no = 0
video_capture = cv2.VideoCapture(camera_no)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

# MediaPipe Hands初期化
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,  # 最大手検出数：1
    min_detection_confidence=0.5,  # 検出信頼度閾値：0.5
    min_tracking_confidence=0.5  # トラッキング信頼度閾値：0.5
)

# 人差指のID
ID_FINGER_TIP = 9
ID_FINGER_TIP_2 = 12

# 人差指の指先の座標履歴を保持するための変数
history_length = 16
point_history = deque(maxlen=history_length)

while video_capture.isOpened():
    # カメラ画像取得
    ret, frame = video_capture.read()
    if ret is False:
        break

    # 鏡映しになるよう反転
    frame = cv2.flip(frame, 1)

    # MediaPipeで扱う画像は、OpenCVのBGRの並びではなくRGBのため変換
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 画像をリードオンリーにしてHands検出処理実施
    rgb_image.flags.writeable = False
    hands_results = hands.process(rgb_image)
    rgb_image.flags.writeable = True

    # 有効なランドマークが検出された場合、ランドマークを描画
    if hands_results.multi_hand_landmarks:
        for hand_landmarks in hands_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks,
                                      mp_hands.HAND_CONNECTIONS)

            # ランドマーク座標の計算
            landmark_list = calc_landmark_list(rgb_image, hand_landmarks)
            # 人差指の指先座標を履歴に追加
            point_history.append(landmark_list[ID_FINGER_TIP])
            point_history.append(landmark_list[ID_FINGER_TIP_2])

    # ディスプレイ表示
    frame = draw_point_history(frame, point_history)
    cv2.imshow('test', frame)

    # キー入力(ESC:プログラム終了)
    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break

# リソースの解放
video_capture.release()
hands.close()
cv2.destroyAllWindows()
