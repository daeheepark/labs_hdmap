# Labs HDMap

## NusToolkit 및 DatasetQ10 사용

### 1. Setup

1) NuScenes 로그인 (https://www.nuscenes.org)
2) 상단 다운로드 페이지 --> Full dataset --> Trainval에서 Metadata 다운로드

### 2. Usage

1) 메타데이터로부터 trajectory 데이터 파싱

```python
from pkyutils import NusToolkit

toolkit = NusToolkit(root='../nus_dataset/original_small/v1.0-mini', version='v1.0-mini', load_dir='../nus_dataset')
toolkit.save_dataset()

idx = 280
past, past_len, future, future_len, agent_mask, vel, pos, map_img, prior, idx = toolkit[idx][0]
```

- 옵션 설명:
  - root: 원본 데이터 경로
  - version: 데이터셋 버전 (v1.0-mini 또는 v1.0-trainval)
  - load_dir: 저장하고자 하는 경로

2) 데이터셋 불러오기

```python
import numpy as np
import torch
import matplotlib.pyplot as plt
from pkyutils import DatasetQ10

dataset = DatasetQ10(version='v1.0-trainval', load_dir='../nus_dataset', data_partition='train',
                     shuffle=False, val_ratio=0.3, data_type='real', min_angle=np.deg2rad(20), max_angle=None)
```

- 옵션 설명:
  - version: 데이터셋 버전 (v1.0-mini 또는 v1.0-trainval)
  - load_dir: 불러오기 위한 데이터셋 경로
  - data_partition: train / val / test
  - val_ratio: validation dataset 비율
  - data_type: real / fake (fake는 주행유도선 기반 가상 데이터)
  - min_angle, max_angle: 불러오고자 하는 데이터셋의 최소/최대 주행각도 (default: None)



---

## Training & Evaluation

### 1. Training

```python
python main_pky.py \
--tag 'anything to remember' --model_type 'AttGlobal_Scene_CAM_NFDecoder' --dataset 'nuscenes' \
--version 'v1.0-trainval' or 'v1.0-mini' --data_type 'real' or 'fake' \
--batch_size 3 (dynamic batch size) --num_epochs 100 --gpu_devices 0 --agent_embed_dim 128 \
--ploss_type 'mseloss' or 'map' \
--beta 0.1 --batch_size 3 --min_angle 0.001745 (radian) --max_angle None (None for no limitation)
```



### 2. Evaluation

```python
python main_pky.py \
--version 'v1.0-trainval' or 'v1.0-mini' --data_type 'real' or 'fake' \
--ploss_type 'mseloss' or 'map' \
--beta 0.1 --batch_size 3 (dynamic batch size) --gpu_devices 0 \
--min_angle 0.001745 (radian) --max_angle None (None for no limitation) \
--test_times 3 (total test times) \
--test_ckpt (model weights path) \
--test_dir (path to save evaluation results)
```



### 3. Testing (Visualization)

```python
python main_pky.py \
--version 'v1.0-trainval' or 'v1.0-mini' --data_type 'real' or 'fake' \
--ploss_type 'mseloss' or 'map' \
--beta 0.1 --batch_size 1 --gpu_devices 0 \
--min_angle 0.001745 (radian) --max_angle None (None for no limitation) \
--test_ckpt (model weights path) \
--test_dir (path to save visuzalized images) --viz
```

