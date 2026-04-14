"""
Wall → Crack 2단계 Segmentation 파이프라인
  1단계: 벽 세그멘테이션 모델 (YOLOv11n-seg) 로 벽 영역 감지
  2단계: 벽 영역만 잘라서 크랙 세그멘테이션 모델 (YOLOv8n-seg) 로 크랙 감지
 
폴더 내 전체 이미지 처리 + CSV 요약 저장
"""
 
import cv2
import csv
import numpy as np
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
 
 
# ── 설정 ────────────────────────────────────────────────────────────────────
WALL_MODEL   = r"./model_file/wall_segment_ver13.pt"
CRACK_MODEL  = r"./model_file/crack_seg_lab.pt" # 버전 1이 더 성과가 좋음
IMAGE_DIR    = r"./test_images"   # ← 폴더 경로
 
WALL_CONF    = 0.25
CRACK_CONF   = 0.25
IOU          = 0.45
 
WALL_COLOR   = (0, 255, 0)       # 벽 마스크 색상 (BGR) - 초록
CRACK_COLOR  = (0, 0, 255)       # 크랙 마스크 색상 (BGR) - 빨강
MASK_ALPHA   = 0.4
OUTPUT_DIR   = r"./results_ver13"       # ← 결과 저장 폴더
 
IMAGE_EXTS   = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
# ────────────────────────────────────────────────────────────────────────────
 
 
def load_models():
    """벽/크랙 모델 로드"""
    for path in [WALL_MODEL, CRACK_MODEL]:
        if not Path(path).exists():
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {path}")
 
    print(f"[INFO] 벽 모델 로드  : {WALL_MODEL}")
    wall_model = YOLO(WALL_MODEL)
 
    print(f"[INFO] 크랙 모델 로드: {CRACK_MODEL}")
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
 
    combined_mask = np.zeros((h, w), dtype=np.uint8)
    if result.masks is None:
        return combined_mask
 
    for mask in result.masks.data.cpu().numpy():
        mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        combined_mask = np.maximum(combined_mask, (mask_resized > 0.5).astype(np.uint8))
 
    return combined_mask
 
 
def detect_cracks_in_wall(img: np.ndarray, wall_mask: np.ndarray, crack_model: YOLO):
    """
    2단계: 벽 마스크 영역만 남긴 이미지로 크랙 감지
    벽 마스크 안에 있는 크랙만 유효로 처리
    """
    masked_img = img.copy()
    masked_img[wall_mask == 0] = 0
 
    results = crack_model.predict(
        source=masked_img, conf=CRACK_CONF, iou=IOU,
        retina_masks=True, verbose=False
    )
    result = results[0]
 
    if result.masks is None:
        return [], []
 
    h, w = img.shape[:2]
    valid_masks = []
    valid_boxes = []
 
    for i, mask in enumerate(result.masks.data.cpu().numpy()):
        mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        binary_mask = (mask_resized > 0.5).astype(np.uint8)
 
        overlap   = np.logical_and(binary_mask, wall_mask).sum()
        crack_area = binary_mask.sum()
        if crack_area > 0 and (overlap / crack_area) >= 0.5:
            valid_masks.append(binary_mask)
            if result.boxes is not None and i < len(result.boxes):
                valid_boxes.append(result.boxes[i])
 
    return valid_masks, valid_boxes
 
 
def visualize_and_save(image_path: str, img: np.ndarray,
                       wall_mask: np.ndarray,
                       crack_masks: list, crack_boxes: list) -> str:
    """결과 시각화: 벽(초록) + 크랙(빨강) 오버레이 후 저장"""
    save_dir = Path(OUTPUT_DIR) / "annotated"
    save_dir.mkdir(parents=True, exist_ok=True)
 
    overlay    = img.copy()
    result_img = img.copy()
 
    # 벽 마스크 (초록)
    overlay[wall_mask == 1] = WALL_COLOR
 
    # 크랙 마스크 (빨강)
    for i, mask in enumerate(crack_masks):
        overlay[mask == 1] = CRACK_COLOR
 
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result_img, contours, -1, CRACK_COLOR, 2)
 
        if i < len(crack_boxes):
            conf_score = float(crack_boxes[i].conf[0])
            x1, y1, x2, y2 = map(int, crack_boxes[i].xyxy[0])
            cv2.putText(result_img, f"crack {conf_score:.2f}",
                        (x1, max(y1 - 8, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, CRACK_COLOR, 2)
 
    blended = cv2.addWeighted(overlay, MASK_ALPHA, result_img, 1 - MASK_ALPHA, 0)
 
    # 상태 텍스트
    wall_pct  = wall_mask.sum() / wall_mask.size * 100
    cv2.putText(blended,
                f"Wall: {wall_pct:.1f}%  |  Cracks: {len(crack_masks)}",
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
 
    # 균열 비율 텍스트
    if wall_mask.sum() > 0 and crack_masks:
        crack_total  = sum(m.sum() for m in crack_masks)
        crack_pct    = crack_total / wall_mask.sum() * 100
        color        = (0, 0, 255) if crack_pct > 5.0 else (0, 200, 0)
        cv2.putText(blended,
                    f"Crack ratio (in wall): {crack_pct:.2f}%",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
 
    # 범례
    cv2.rectangle(blended, (10, 85),  (30, 105), WALL_COLOR,  -1)
    cv2.putText(blended, "Wall",  (35, 100),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    cv2.rectangle(blended, (90, 85), (110, 105), CRACK_COLOR, -1)
    cv2.putText(blended, "Crack", (115, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
 
    stem      = Path(image_path).stem
    suffix    = Path(image_path).suffix
    save_path = save_dir / f"{stem}_result{suffix}"
    cv2.imwrite(str(save_path), blended)
    return str(save_path)
 
 
def calc_stats(wall_mask: np.ndarray, crack_masks: list) -> dict:
    """이미지 1장 통계 계산"""
    wall_px    = int(wall_mask.sum())
    total_px   = int(wall_mask.size)
    wall_pct   = round(wall_px / total_px * 100, 4) if total_px > 0 else 0.0
 
    crack_px   = int(sum(m.sum() for m in crack_masks)) if crack_masks else 0
    crack_pct  = round(crack_px / wall_px * 100, 4) if wall_px > 0 else 0.0
 
    return {
        "n_cracks":   len(crack_masks),
        "wall_px":    wall_px,
        "wall_pct":   wall_pct,
        "crack_px":   crack_px,
        "crack_pct":  crack_pct,         # 벽 면적 대비 균열 비율 (핵심 지표)
    }
 
 
def save_csv(rows: list, total: int, crack_detected: int):
    """전체 결과 summary.csv 저장"""
    csv_path = Path(OUTPUT_DIR) / "summary.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
 
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
 
        # 메타
        writer.writerow(["실행 시각",       datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
        writer.writerow(["벽 모델",          WALL_MODEL])
        writer.writerow(["크랙 모델",        CRACK_MODEL])
        writer.writerow(["이미지 폴더",      IMAGE_DIR])
        writer.writerow(["총 이미지",        total])
        writer.writerow(["균열 감지 이미지", crack_detected])
        writer.writerow([])
 
        # 헤더
        writer.writerow([
            "파일명",
            "균열 감지 수",
            "벽 면적(px)",
            "벽 비율(%)",
            "균열 면적(px)",
            "벽 대비 균열 비율(%)",
            "상태",
        ])
 
        # 데이터
        writer.writerows(rows)
 
    print(f"\n  CSV 저장 완료: {csv_path}")
    return str(csv_path)
 
 
def process_folder():
    """폴더 내 전체 이미지 처리"""
 
    # 이미지 목록
    img_dir   = Path(IMAGE_DIR)
    img_paths = sorted([
        p for p in img_dir.rglob("*")
        if p.suffix.lower() in IMAGE_EXTS
    ])
 
    if not img_paths:
        print(f"[오류] 이미지를 찾을 수 없습니다: {img_dir}")
        return
 
    total = len(img_paths)
    print(f"[시작] 총 {total}장 처리 시작")
    print(f"       결과 저장: {Path(OUTPUT_DIR).absolute()}\n")
 
    # 모델 로드
    wall_model, crack_model = load_models()
 
    rows           = []
    crack_detected = 0
 
    for idx, img_path in enumerate(img_paths, 1):
        print(f"  [{idx:>4}/{total}] {img_path.name}")
 
        # 이미지 로드
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"           ↳ 읽기 실패, 건너뜁니다.")
            rows.append([img_path.name, "-", "-", "-", "-", "-", "읽기 실패"])
            continue
 
        # 1단계: 벽 감지
        wall_mask = get_wall_mask(img, wall_model)
 
        if wall_mask.sum() == 0:
            print(f"           ↳ 벽 미감지, 크랙 감지 생략")
            rows.append([img_path.name, 0, 0, 0.0, 0, 0.0, "벽 미감지"])
            continue
 
        # 2단계: 크랙 감지
        crack_masks, crack_boxes = detect_cracks_in_wall(img, wall_mask, crack_model)
 
        # 통계 계산
        stats = calc_stats(wall_mask, crack_masks)
 
        # 결과 이미지 저장
        visualize_and_save(str(img_path), img, wall_mask, crack_masks, crack_boxes)
 
        # 상태 결정
        if stats["n_cracks"] > 0:
            crack_detected += 1
            status = f"균열 {stats['crack_pct']:.2f}%"
        else:
            status = "균열 없음"
 
        print(f"           ↳ 벽 {stats['wall_pct']:.1f}%  |  "
              f"크랙 {stats['n_cracks']}개  |  벽 대비 균열 {stats['crack_pct']:.2f}%")
 
        rows.append([
            img_path.name,
            stats["n_cracks"],
            stats["wall_px"],
            stats["wall_pct"],
            stats["crack_px"],
            stats["crack_pct"],
            status,
        ])
 
    # CSV 저장
    csv_path = save_csv(rows, total, crack_detected)
 
    # 최종 요약
    print(f"\n{'─'*55}")
    print(f"  완료         : {total}장")
    print(f"  균열 감지    : {crack_detected}장  ({crack_detected/total*100:.1f}%)")
    print(f"  결과 이미지  : {Path(OUTPUT_DIR) / 'annotated'}")
    print(f"  CSV 요약     : {csv_path}")
    print(f"{'─'*55}\n")
 
 
if __name__ == "__main__":
    process_folder()