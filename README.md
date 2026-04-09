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



#### Classification(3개의 모델 중 고민 중)
 - EfficientNet-B0
 - ResNet18
 - MobileNetV3