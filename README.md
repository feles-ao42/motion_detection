# motion_detection

### install
```shell
pip3 install -r requiment.txt
```

### mediapipe動作test
```shell
python3 test.py
```

### データセット作成（作成済みモデルの場合は不要）
```shell
python3 train.py --gesture_id=1 --time=10
#0は静止,1はパーで手を振る,2はグーで手を振る
```

### 作成したデータセットを用いた学習（作成済みモデルの場合は不要）
```shell
python3 lern.py
```

### モーション検知本体
```shell
python3 main.py
```