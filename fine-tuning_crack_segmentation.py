from ultralytics import YOLO

def main():
    # 1. 모델 로드 (상황에 따라 yolo11s-seg.pt, yolo11m-seg.pt 등 선택)
    model = YOLO("runs/crack/train_v1/weights/best.pt")

    # 2. 파인튜닝 학습 진행
    results = model.train(
        data="datasets/data.yaml",               # 데이터셋 설정 파일
        epochs=50,                     # 충분한 에폭 설정 (Patience로 조기 종료 대비)
        imgsz=1280,                      # 입력 해상도 (균열 등 미세 탐지시 1280까지 고려 가능하나 VRAM 주의)
        batch=2,                        # RTX 4060 Ti의 VRAM(8GB/16GB)을 고려한 배치 사이즈 (OOM 발생 시 4로 하향)
        device=0,                       # GPU 할당
        project="cctv_monitoring_cv",   # 프로젝트 명칭
        name="fine-tuning_run",    # 실험 명칭

        # --- 프리징 (Freezing) ---
        freeze = 10, # 백본(Backbone)의 첫 10개 레이어를 고정하여 학습(가중치 동결) (초기에는 0으로 시작하여 성능과 VRAM 상황에 따라 조절)

        # --- warmup (Warmup) ---
        warmup_epochs=3,               # 초기 학습률을 낮게 시작하여 점진적으로 증가시키는 워밍업 기간 (과적합 방지 및 안정적인 수렴 유도)
        warmup_momentum=0.8,           # 워밍업 동안 모멘텀 설정
        warmup_bias_lr=0.1,            # 워밍업 동안 바이어스 학습률 설정
       
        # --- 최적화 (Optimization) ---
        amp=True,                       # RTX 40 시리즈 텐서 코어를 활용한 Mixed Precision 학습 (속도 증가, VRAM 절약)
        optimizer="AdamW",              # 복잡한 패턴 학습에 유리한 옵티마이저
        lr0=0.001,                      # 초기 학습률
        lrf=0.01,                       # 최종 학습률 비율
        weight_decay=0.0005,            # 과적합 방지
        patience=25,                    # 성능 개선이 없을 시 조기 종료
       
        # --- 데이터 증강 (Data Augmentation) ---
        # 주의: 라이브러리 버전에 따라 에러를 유발하는 blur, box_noise 파라미터는 완전히 제외함.
        hsv_h=0.015,                    # 색상(Hue) 변환
        hsv_s=0.7,                      # 채도(Saturation) 변환
        hsv_v=0.4,                      # 명도(Value) 변환
        degrees=0.0,                    # 회전 (CCTV 등 고정 앵글인 경우 0 유지 추천)
        translate=0.1,                  # 이미지 이동
        scale=0.5,                      # 크기 조절
        flipud=0.0,                     # 상하 반전 (상하 뷰가 고정된 모니터링 환경에서는 0)
        fliplr=0.5,                     # 좌우 반전 (일반적으로 유효)
        mosaic=1.0,                     # 모자이크 증강 (작은 객체 탐지에 매우 효과적)
        mixup=0.1,                      # 이미지 믹스업
        erasing=0.4,                    # Random Erasing (가려짐 현상에 대한 강건함 확보)
    )
   
    print("Training Complete. Best model saved at:", results.save_dir)

if __name__ == '__main__':
    # Windows 환경에서 멀티프로세싱(DataLoader) 충돌을 방지하기 위한 안전장치
    main()