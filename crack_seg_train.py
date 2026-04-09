"""
Crack Segmentation - 기본 학습 코드
Dataset: crack-seg (Ultralytics)
Model: YOLOv11n-seg (pretrained)
"""

# pip install ultralytics

from ultralytics import YOLO

# ── 모델 로드 ──────────────────────────────────────────────────────
# nano / small / medium 중 선택 (n < s < m, 크기/정확도 트레이드오프)
model = YOLO("yolo11n-seg.pt")

# ── 학습 ──────────────────────────────────────────────────────────
results = model.train(
    data="crack-seg.yaml",       # Ultralytics 내장 데이터셋 (자동 다운로드)
    epochs=100,
    imgsz=640,
    batch=16,                    # GPU 메모리 부족 시 8로 낮추기
    device=0,                    # GPU 0번 사용 (CPU는 "cpu")
    project="runs/crack",
    name="train_v1",
    exist_ok=True,

    # ── 데이터 증강 ──────────────────────────────────────────────
    hsv_h=0.015,      # Hue 변화 (색조)
    hsv_s=0.7,        # Saturation 변화 (채도)
    hsv_v=0.4,        # Value 변화 (명도)
    degrees=10.0,     # 회전 (-10 ~ +10도)
    translate=0.1,    # 이동 (이미지 크기의 10%)
    scale=0.5,        # 스케일 변화 (0.5 ~ 1.5배)
    shear=2.0,        # 기울이기
    flipud=0.3,       # 상하 반전 확률
    fliplr=0.5,       # 좌우 반전 확률
    mosaic=1.0,       # Mosaic 증강 (4장 합치기)
    copy_paste=0.3,   # Copy-Paste 증강 (segmentation에 유용)
    # ─────────────────────────────────────────────────────────────
)

print("학습 완료:", results)


# ── 검증 ──────────────────────────────────────────────────────────
metrics = model.val()
print(f"mAP50      : {metrics.seg.map50:.4f}")
print(f"mAP50-95   : {metrics.seg.map:.4f}")


# ── 추론 (학습 후 테스트) ─────────────────────────────────────────
# model.predict(
#     source="path/to/test/images",
#     save=True,
#     conf=0.25,
#     iou=0.45,
# )
