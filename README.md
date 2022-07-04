# YOLO : 🐯Yonsei gOnna Lose fOrever🐯 
- YOLO V1 모델을 구현한 결과물 레포지토리입니다.
  - Training 과정의 디테일에 집중했습니다.
  - 고연전 오대빵을 기원합니다.

# 설명
- 제 작업물은 `./loggerJK` 폴더 안에 있습니다.
  - `YOLO_singleLoss.ipynb`
    - 배치 단위 처리를 지원하지 않는 버전의 YOLO입니다.
  - `YOLO_batchLoss.ipynb`
    - 배치 단위 처리를 지원하도록 개선한 버전의 YOLO입니다.
  - `YOLO_batchLoss_trainval.ipynb`
    - Training Set만으로는 학습이 어려워 Training / Validation Set 모두 학습에 이용한 노트북입니다.
- 학습한 모델의 Inference 결과물은 `./loggerJK/Model Test` 폴더 안에 있습니다.
  - `./loggerJK/Model Test/model_test.ipynb`
  - Inference 과정 중 mAP 계산, Non-Maximum Suppression과 같은 부분들은 구현되어 있지 않았습니다.

# 모델 설명
- Base Model : Vision Transformer
  - `vit_base_patch32_384` from `timm`
- `input_size` : $384 \times 384$
- `learning_rate` : 1e-5 (fixed)
- `epoch` : 70
  - 이 외의 기타 Training에 관련된 수학적 디테일들은 논문과 동일하거나, 최대한 유사하도록 구현했습니다.
