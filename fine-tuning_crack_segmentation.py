from ultralytics import YOLO
from pathlib import Path

def main():
    # =============================================
    # 1단계: 현장 데이터로 초기 fine-tuning
    # 백본을 강하게 고정하고 헤드(Head)만 집중 학습
    # =============================================
    model = YOLO("runs/crack/train_v1/weights/best.pt")

    results = model.train(
        data="datasets/data.yaml",      # 현장 데이터 yaml 확인 필수
        epochs=40,                       # 300장 기준 40으로 충분 (patience가 조기종료)
        imgsz=1280,
        batch=2,
        device=0,
        project="cctv_monitoring_cv",
        name="finetune_field_v1",
        cache=True,                      # 300장은 RAM에 캐싱 가능, 속도 향상

        # --- 프리징 ---
        freeze=15,                       # 300장 소량이므로 백본 더 강하게 고정

        # --- Warmup ---
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,

        # --- 최적화 ---
        amp=True,
        optimizer="AdamW",
        lr0=0.0005,                      # 이미 학습된 모델이므로 낮게 시작
        lrf=0.01,
        weight_decay=0.0005,
        patience=20,
        close_mosaic=10,                 # 마지막 10 epoch mosaic 끄기 (수렴 안정화)

        # --- 데이터 증강 (300장 기준 조정) ---
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,                     # CCTV 고정 앵글 유지
        translate=0.1,
        scale=0.5,
        flipud=0.0,
        fliplr=0.5,
        mosaic=0.5,                      # 1.0 → 0.5, crack 형태 왜곡 방지
        mixup=0.05,                      # 0.1 → 0.05, 소량 데이터에서 완화
        erasing=0.3,                     # 0.4 → 0.3
    )

    print("1단계 완료. 저장 경로:", results.save_dir)


    # =============================================
    # 2단계: 더 세밀한 조정
    # freeze 낮추고 lr도 더 낮게
    # =============================================
    best_model_path = Path(results.save_dir) / "weights" / "best.pt"  # 동적 경로
    model = YOLO(str(best_model_path))

    model.train(
        data="datasets/data.yaml",
        epochs=25,                       # 2단계는 짧게
        imgsz=1280,
        batch=2,
        device=0,
        project="cctv_monitoring_cv",
        name="finetune_field_v2",
        exist_ok=True,
        cache=True,

        freeze=5,                        # 백본 일부만 고정, 세밀한 조정 허용
        
        warmup_epochs=2,                 # 2단계는 워밍업 짧게
        warmup_momentum=0.8,
        warmup_bias_lr=0.05,

        amp=True,
        optimizer="AdamW",
        lr0=0.0001,                      # 2단계는 더 낮게
        lrf=0.01,
        weight_decay=0.0005,
        patience=15,
        close_mosaic=8,

        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        flipud=0.0,
        fliplr=0.5,
        mosaic=0.3,                      # 2단계에서 더 낮춤
        mixup=0.0,                       # 2단계에서는 끄기
        erasing=0.2,
    )

if __name__ == '__main__':
    main()