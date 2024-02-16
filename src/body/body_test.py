# -*- coding: utf-8 -*-
import cv2
import mediapipe as mp
from collections import deque

# ポーズの画像上の位置を算出する関数
def calc_pose_landmarks(image, landmarks):
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
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# MediaPipe Pose初期化
mp_pose = mp.solutions.pose

pose = mp_pose.Pose(
    min_detection_confidence=0.5,  # 検出信頼度閾値：0.5
    min_tracking_confidence=0.5  # トラッキング信頼度閾値：0.5
)

# トラック対象の設定
ID_SHOULDER = 11  # 右肩
ID_HAND = 20  # 右手首

# 体の部位の座標履歴を保持するための変数
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

    # 画像をリードオンリーにしてPose検出処理実施
    rgb_image.flags.writeable = False
    pose_results = pose.process(rgb_image)
    rgb_image.flags.writeable = True

    # 有効なランドマークが検出された場合、ランドマークを描画
    if pose_results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # ポーズのランドマーク座標の計算
        pose_landmark_list = calc_pose_landmarks(rgb_image, pose_results.pose_landmarks)
        # 肩と腰の座標を履歴に追加
        point_history.append(pose_landmark_list[ID_SHOULDER])
        point_history.append(pose_landmark_list[ID_HAND])

    # ディスプレイ表示
    frame = draw_point_history(frame, point_history)
    cv2.imshow('test', frame)

    # キー入力(ESC:プログラム終了)
    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break

# リソースの解放
video_capture.release()
pose.close()
cv2.destroyAllWindows()
