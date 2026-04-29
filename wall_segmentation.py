"""
벽이 감지된 이미지만 별도 폴더로 추출 + 마스크 오버레이 시각화 저장
- 벽 세그멘테이션 모델(YOLOv11n-seg)로 벽 감지 여부 판단
- 감지된 이미지에 마스크/컨투어 오버레이 그려서 ./wall_detected 폴더에 저장
"""
 
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
 
# ── 설정 ──────────────────────────────────────────────────────────────────
WALL_MODEL  = r"./model_file/wall_segment_ver13.pt"
IMAGE_DIR   = r"./farmer"
OUTPUT_DIR  = r"./wall_detected_farmer"
WALL_CONF   = 0.30
IOU         = 0.45
MASK_ALPHA  = 0.4
WALL_COLOR  = (0, 255, 0)          # 초록 마스크
IMAGE_EXTS  = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
# ──────────────────────────────────────────────────────────────────────────
 
 
def draw_wall_overlay(img: np.ndarray, result, h: int, w: int) -> np.ndarray:
    """벽 마스크 + 컨투어 + 라벨 오버레이를 그린 이미지 반환"""
    overlay    = img.copy()
    result_img = img.copy()
 
    for i, mask in enumerate(result.masks.data.cpu().numpy()):
        mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        binary       = (mask_resized > 0.5).astype(np.uint8)
 
        # 반투명 마스크
        overlay[binary == 1] = WALL_COLOR
 
        # 컨투어
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result_img, contours, -1, WALL_COLOR, 2)
 
        # 박스 + confidence 라벨
        if result.boxes is not None and i < len(result.boxes):
            conf = float(result.boxes[i].conf[0])
            x1, y1, x2, y2 = map(int, result.boxes[i].xyxy[0])
            cv2.putText(result_img, f"wall {conf:.2f}",
                        (x1, max(y1 - 8, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, WALL_COLOR, 2)
 
    blended = cv2.addWeighted(overlay, MASK_ALPHA, result_img, 1 - MASK_ALPHA, 0)
 
    # 상단 요약 텍스트
    wall_count = len(result.masks.data)
    cv2.putText(blended,
                f"Walls detected: {wall_count}",
                (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
 
    # 범례
    cv2.rectangle(blended, (10, 45), (30, 65), WALL_COLOR, -1)
    cv2.putText(blended, "Wall", (35, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)
 
    return blended
 
 
def main():
    img_dir = Path(IMAGE_DIR)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
 
    img_paths = sorted([
        p for p in img_dir.rglob("*")
        if p.suffix.lower() in IMAGE_EXTS
    ])
 
    if not img_paths:
        print(f"[오류] 이미지를 찾을 수 없습니다: {img_dir}")
        return
 
    total = len(img_paths)
    print(f"[시작] 총 {total}장 검사")
    print(f"[모델] {WALL_MODEL}")
    print(f"[출력] {out_dir.absolute()}\n")
 
    wall_model = YOLO(WALL_MODEL)
    detected_count = 0
 
    for idx, img_path in enumerate(img_paths, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [{idx:>4}/{total}] {img_path.name}  →  읽기 실패, 건너뜀")
            continue
 
        h, w = img.shape[:2]
 
        results = wall_model.predict(
            source=img,
            conf=WALL_CONF,
            iou=IOU,
            retina_masks=True,
            verbose=False,
        )
        result = results[0]
 
        wall_count = len(result.masks.data) if result.masks is not None else 0
 
        if wall_count > 0:
            blended = draw_wall_overlay(img, result, h, w)
 
            # 파일명: 원본명_wall_result.확장자
            stem     = img_path.stem
            suffix   = img_path.suffix
            out_path = out_dir / f"{stem}_wall_result{suffix}"
            cv2.imwrite(str(out_path), blended)
 
            detected_count += 1
            print(f"  [{idx:>4}/{total}] {img_path.name}  →  ✅ 벽 {wall_count}개 감지 → {out_path.name}")
        else:
            print(f"  [{idx:>4}/{total}] {img_path.name}  →  ❌ 벽 미감지, 제외")
 
    print(f"\n{'─'*55}")
    print(f"  전체       : {total}장")
    print(f"  벽 감지    : {detected_count}장  ({detected_count / total * 100:.1f}%)")
    print(f"  제외       : {total - detected_count}장")
    print(f"  저장 위치  : {out_dir.absolute()}")
    print(f"{'─'*55}\n")
 
 
if __name__ == "__main__":
    main()