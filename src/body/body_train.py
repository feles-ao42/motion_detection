import csv
import time
import argparse
from collections import deque

import cv2
import mediapipe as mp

# コマンドライン引数取得
parser = argparse.ArgumentParser()
parser.add_argument("--time", type=int, default=10)
parser.add_argument("--gesture_id", type=int, default=0)
args = parser.parse_args()


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


# CSVファイルに座標履歴を保存する関数
def logging_csv(gesture_id, csv_path, width, height, point_history_list):
    with open(csv_path, 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow([gesture_id, width, height, *point_history_list])
    return


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

# CSVファイル保存先
csv_path = '../data/body_point_history.csv'

start_time = time.time()
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
        point_history_list = [item for sublist in point_history for item in sublist]
        logging_csv(args.gesture_id, csv_path, frame_width, frame_height, point_history_list)

    # ディスプレイ表示
    frame = draw_point_history(frame, point_history)
    cv2.imshow('full_body_detection', frame)

    # キー入力(ESC:プログラム終了)
    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break

    if (time.time() - start_time) > args.time:
        break

# リソースの解放
video_capture.release()
pose.close()
cv2.destroyAllWindows()
