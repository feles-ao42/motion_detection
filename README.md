# motion_detection
ミニ実験用です。
mediapipeを用いた、モーション検知ができます。

### install
```shell
pip install -r ./env/requirements.txt
```

### mediapipe動作test
```shell
python3 src/body/body_test.py
```

### データセット作成（作成済みモデルの場合は不要）
```shell
python3 src/body/body_train.py --gesture_id=1 --time=10
#0は静止,1はパーで手を振る,2はグーで手を振る
```

### 作成したデータセットを用いた学習（作成済みモデルの場合は不要）
```shell
python3 src/body/body_lern.py
```

### モーション検知本体
```shell
python3 src/body/body_main.py
```