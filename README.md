# YOLOv26n-seg OpenDatasets training

처음 학습은 먼저 원래 준 코드와 데이터셋으로 학습 진행

그 후 imagsz와 데이터 증강을 하여 학습을 진행 후 실제 현장에 사진을 가지고 Fine-tuning 진행 예정
학습 진행 결과는 따로 남겨둠



## 후처리 방법 고민


### 문제1. 크랙 감지다 보니 객체가 너무 작은 문제가 발생
참조 : "Small Object Detection Based on Deep Learning for Remote Sensing: A Comprehensive Review" Xuan Wang1 ,AoranWang1,Jinglei Yi 1, Yongchao Song 1 and Abdellah Chehri 2,*


### 문제2. 후처리 비, 안개 같은 변수 발생 시 객체 인식률 하락

모폴로지 연산
  - Closing 먼저 (끊어진 균열 연결)
  - Opening 나중에, 커널 3×3 + 십자형으로 얇은 균열 보존

기하학적 필터
  - 가로세로 비율 1:3 이하 제거
  - 면적 너무 작은 것 제거
  - 그 후 Classification 모델에 전달


### 문재3. 오탐 발생률 높음(다양한 해결방안 구안 중)
 - 데이터 증가(가장 원초적인 방법)
 - 오탐 라벨링
 - 단계(Detection -> classification -> Segmentation)

---

#### Segmentation, Detection 다양한 모델을 사용하였으나 현장 상황을 고려하여 yolov11n-Seg으로 진행(벽 감지)
 - RT-DETR (Detection)
 - yolov11s-Seg
 - yolov26n-Seg
 - yolov26s-Seg
 - U-Net++ (Segmentation)
 - Attention U-Net (Segmentation)
 - TransUNet (Segmentation)

---


#### Classification(3개의 모델 중 고민 중)
 - EfficientNet-B0
 - ResNet18
 - MobileNetV3


---

##### 2026.04.10 crack 실험 결과 Segmentation 인식 잘 됨 하지만 오탐을 줄이기 위해서 진행했던 벽 Segmentation에서 벽 감지가 안되서 문제가 발생
 - 위 문제를 해결하기 위해 벽 감지 학습 다시 진행
 - 데이터셋 추가(찾아보기)
 

---

##### 2026.04.14 crack 오탐을 줄이기 위해 classification 사용 하지만 아직 크랙이 감지가 안되는게 몇개 있어 실제 사진으로 fine-tuning wall_seg imgsz 448로 바꿈 오히려 wall_seg 잘됨 crack은 그대로 1280 유지
 - wall_seg_crack_seg_crack_classification.py은 전체 면적분에 크랙의 비율(픽셀)이다.
 - wall_seg_crack_seg_crack_classification_ver2.py은 크랙이 나온 Segmentation(wall) 된 범위 안에서의 크랙 비율(픽셀)이다. 
 - fine-tuning을 사용할 때 백본 얼리고 fine-tuning
 - classification은 결과 나오면 오탐 확인 후 재학습
 - EfficientNet-B0 사용