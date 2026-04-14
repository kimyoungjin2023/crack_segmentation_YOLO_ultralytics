"""
Wall → Crack 2단계 Segmentation 파이프라인
  1단계: 벽 세그멘테이션 모델 (YOLOv11n-seg) 로 벽 영역 감지
  2단계: 벽 영역만 잘라서 크랙 세그멘테이션 모델 (YOLOv8n-seg) 로 크랙 감지
"""
 
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
 
 
# ── 설정 ────────────────────────────────────────────────────────────────────
WALL_MODEL   = r"./model_file/wall_segment_ver13.pt"    # ← 벽 세그멘테이션 모델 경로
CRACK_MODEL  = r"./model_file/crack_seg_lab.pt"   # ← 크랙 세그멘테이션 모델 경로
TEST_IMAGE   = "test_camera.JPG"        # ← 테스트 이미지 경로
 
WALL_CONF    = 0.25              # 벽 감지 confidence threshold
CRACK_CONF   = 0.25              # 크랙 감지 confidence threshold
IOU          = 0.45              # NMS IoU threshold
 
WALL_COLOR   = (0, 255, 0)       # 벽 마스크 색상 (BGR) - 초록색
CRACK_COLOR  = (0, 0, 255)       # 크랙 마스크 색상 (BGR) - 빨간색
MASK_ALPHA   = 0.4               # 마스크 투명도
OUTPUT_DIR   = "output"          # 결과 저장 폴더
# ────────────────────────────────────────────────────────────────────────────
 
 
def load_models():
    """벽/크랙 모델 로드"""
    for path in [WALL_MODEL, CRACK_MODEL]:
        if not Path(path).exists():
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {path}")
 
    print(f"[INFO] 벽 모델 로드 (YOLOv11n-seg): {WALL_MODEL}")
    wall_model = YOLO(WALL_MODEL)
 
    print(f"[INFO] 크랙 모델 로드 (YOLOv8n-seg): {CRACK_MODEL}")
    crack_model = YOLO(CRACK_MODEL)
 
    print("[INFO] 모델 로드 완료\n")
    return wall_model, crack_model
 
 
def get_wall_mask(img: np.ndarray, wall_model: YOLO) -> np.ndarray:
    """
    1단계: 벽 세그멘테이션 → 통합 벽 마스크 반환
    반환값: (H, W) 이진 마스크 (벽=1, 배경=0)
    """
    h, w = img.shape[:2]
    results = wall_model.predict(
        source=img, conf=WALL_CONF, iou=IOU,
        retina_masks=True, verbose=False
    )
    result = results[0]
 
    # 벽 마스크가 없으면 빈 마스크 반환
    combined_mask = np.zeros((h, w), dtype=np.uint8)
    if result.masks is None:
        print("[WARN] 벽이 감지되지 않았습니다.")
        return combined_mask
 
    for mask in result.masks.data.cpu().numpy():
        mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        combined_mask = np.maximum(combined_mask, (mask_resized > 0.5).astype(np.uint8))
 
    wall_count = len(result.masks.data)
    print(f"[INFO] 벽 감지 완료: {wall_count}개 영역")
    return combined_mask
 
 
def detect_cracks_in_wall(img: np.ndarray, wall_mask: np.ndarray, crack_model: YOLO):
    """
    2단계: 벽 마스크 영역만 남긴 이미지로 크랙 감지
    - 벽 마스크를 원본 이미지에 적용 후 크랙 모델 입력
    - 크랙 결과 중 벽 마스크 안에 있는 것만 필터링
    """
    # 벽 영역만 남기고 나머지는 검정 처리
    masked_img = img.copy()
    masked_img[wall_mask == 0] = 0
 
    results = crack_model.predict(
        source=masked_img, conf=CRACK_CONF, iou=IOU,
        retina_masks=True, verbose=False
    )
    result = results[0]
 
    if result.masks is None:
        print("[INFO] 크랙이 감지되지 않았습니다.")
        return [], []
 
    h, w = img.shape[:2]
    valid_masks = []
    valid_boxes = []
 
    for i, mask in enumerate(result.masks.data.cpu().numpy()):
        mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        binary_mask = (mask_resized > 0.5).astype(np.uint8)
 
        # 벽 마스크와 겹치는 픽셀 비율 계산 → 50% 이상이면 유효한 크랙
        overlap = np.logical_and(binary_mask, wall_mask).sum()
        crack_area = binary_mask.sum()
        if crack_area > 0 and (overlap / crack_area) >= 0.5:
            valid_masks.append(binary_mask)
            if result.boxes is not None and i < len(result.boxes):
                valid_boxes.append(result.boxes[i])
 
    print(f"[INFO] 크랙 감지 완료: {len(valid_masks)}개 (벽 영역 내)")
    return valid_masks, valid_boxes
 
 
def visualize_and_save(image_path: str, img: np.ndarray,
                       wall_mask: np.ndarray,
                       crack_masks: list, crack_boxes: list) -> str:
    """결과 시각화: 벽(초록) + 크랙(빨강) 오버레이 후 저장"""
    save_dir = Path(OUTPUT_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)
 
    overlay = img.copy()
    result_img = img.copy()
 
    # 벽 마스크 시각화 (초록)
    overlay[wall_mask == 1] = WALL_COLOR
 
    # 크랙 마스크 시각화 (빨강)
    for i, mask in enumerate(crack_masks):
        overlay[mask == 1] = CRACK_COLOR
 
        # 윤곽선
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result_img, contours, -1, CRACK_COLOR, 2)
 
        # confidence 라벨
        if i < len(crack_boxes):
            conf_score = float(crack_boxes[i].conf[0])
            x1, y1, x2, y2 = map(int, crack_boxes[i].xyxy[0])
            cv2.putText(result_img, f"crack {conf_score:.2f}",
                        (x1, max(y1 - 8, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, CRACK_COLOR, 2)
 
    # 블렌딩
    blended = cv2.addWeighted(overlay, MASK_ALPHA, result_img, 1 - MASK_ALPHA, 0)
 
    # 상태 텍스트
    wall_area_pct = (wall_mask.sum() / wall_mask.size * 100)
    cv2.putText(blended, f"Wall area: {wall_area_pct:.1f}%  |  Cracks: {len(crack_masks)}",
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
 
    # 범례
    cv2.rectangle(blended, (10, 50), (30, 70), WALL_COLOR, -1)
    cv2.putText(blended, "Wall", (35, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.rectangle(blended, (90, 50), (110, 70), CRACK_COLOR, -1)
    cv2.putText(blended, "Crack", (115, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
 
    # 저장
    stem = Path(image_path).stem
    suffix = Path(image_path).suffix
    save_path = save_dir / f"{stem}_wall_crack_result{suffix}"
    cv2.imwrite(str(save_path), blended)
    print(f"[INFO] 결과 저장: {save_path}")
    return str(save_path)
 
 
def print_summary(wall_mask: np.ndarray, crack_masks: list):
    """결과 요약"""
    print("\n" + "=" * 45)
    print("           파이프라인 결과 요약")
    print("=" * 45)
    wall_px = wall_mask.sum()
    total_px = wall_mask.size
    print(f"  벽 감지 면적   : {wall_px:,} px ({wall_px/total_px*100:.1f}%)")
    print(f"  크랙 감지 수   : {len(crack_masks)}개")
    if crack_masks:
        crack_total = sum(m.sum() for m in crack_masks)
        crack_in_wall = (crack_total / wall_px * 100) if wall_px > 0 else 0
        print(f"  크랙 총 면적   : {crack_total:,} px")
        print(f"  벽 대비 크랙   : {crack_in_wall:.2f}%")
    print("=" * 45 + "\n")
 
 
def main():
    image_path = TEST_IMAGE
 
    # 이미지 로드
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")
    print(f"[INFO] 이미지 로드: {image_path} ({img.shape[1]}x{img.shape[0]})\n")
 
    # 모델 로드
    wall_model, crack_model = load_models()
 
    # 1단계: 벽 감지
    print("[STEP 1] 벽 세그멘테이션 실행 중...")
    wall_mask = get_wall_mask(img, wall_model)
 
    # 벽이 없으면 종료
    if wall_mask.sum() == 0:
        print("[STOP] 벽이 감지되지 않아 크랙 감지를 건너뜁니다.")
        return
 
    # 2단계: 크랙 감지 (벽 영역 안에서만)
    print("\n[STEP 2] 크랙 세그멘테이션 실행 중...")
    crack_masks, crack_boxes = detect_cracks_in_wall(img, wall_mask, crack_model)
 
    # 시각화 및 저장
    save_path = visualize_and_save(image_path, img, wall_mask, crack_masks, crack_boxes)
 
    # 결과 요약
    print_summary(wall_mask, crack_masks)
    print(f"[DONE] 결과 이미지: {save_path}")
 
 
if __name__ == "__main__":
    main()