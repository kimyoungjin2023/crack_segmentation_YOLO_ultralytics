"""
Wall → Crack → Classification 3단계 Segmentation 파이프라인
  1단계: 벽 세그멘테이션 모델 (YOLOv11n-seg) 로 벽 영역 감지
  2단계: 벽 영역만 잘라서 크랙 세그멘테이션 모델 (YOLOv8n-seg) 로 크랙 감지
  3단계: 크랙 후보 각각을 EfficientNet-B0 분류 모델로 검증 (positive/negative 오탐 필터링)
  → 최종 통과된 크랙만으로 벽 픽셀 대비 크랙 픽셀 비율 계산
 
폴더 내 전체 이미지 처리 + CSV 요약 저장
"""
 
import cv2
import csv
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
 
 
# ── 설정 ────────────────────────────────────────────────────────────────────
WALL_MODEL   = r"./model_file/wall_segment_ver13.pt"
CRACK_MODEL  = r"./model_file/crack_seg_lab.pt"
CLASS_MODEL  = r"./model_file/model.pth"  # EfficientNet-B0 state_dict
 
IMAGE_DIR    = r"./test_images"
 
WALL_CONF    = 0.25
CRACK_CONF   = 0.25
CLASS_CONF   = 0.50    # positive 판정 최소 신뢰도 (0.0 ~ 1.0)
IOU          = 0.45
 
# ※ ImageFolder는 폴더명을 알파벳 순으로 인덱스 할당
#   negative=0, positive=1  (n < p 알파벳 순)
#   폴더 구조가 다르면 아래 값을 맞게 수정하세요
POSITIVE_CLASS_INDEX = 1   # positive 클래스 인덱스
 
# 크랙 bbox 주변 패딩 (분류 모델 입력 크롭 시 여유 픽셀)
CROP_PADDING = 20
 
WALL_COLOR   = (0, 255, 0)
CRACK_COLOR  = (0, 0, 255)
REJECT_COLOR = (128, 128, 128)
MASK_ALPHA   = 0.4
OUTPUT_DIR   = r"./results_ver15"
 
IMAGE_EXTS   = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
 
# EfficientNet 입력 전처리 (학습 코드와 동일하게 맞춤)
CLASSIFY_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
# ────────────────────────────────────────────────────────────────────────────
 
 
def load_class_model(path: str) -> tuple:
    """
    EfficientNet-B0 구조 선언 후 state_dict 로드
    반환: (model, device)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
 
    print(f"[INFO] 분류 모델 device    : {device}")
    return model, device
 
 
def load_models():
    """벽 / 크랙 / 분류 모델 로드"""
    for path in [WALL_MODEL, CRACK_MODEL, CLASS_MODEL]:
        if not Path(path).exists():
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {path}")
 
    print(f"[INFO] 벽 모델 로드        : {WALL_MODEL}")
    wall_model  = YOLO(WALL_MODEL)
 
    print(f"[INFO] 크랙 모델 로드      : {CRACK_MODEL}")
    crack_model = YOLO(CRACK_MODEL)
 
    print(f"[INFO] 분류 모델 로드      : {CLASS_MODEL}")
    class_model, class_device = load_class_model(CLASS_MODEL)
 
    print("[INFO] 모델 로드 완료\n")
    return wall_model, crack_model, class_model, class_device
 
 
# ── 1단계: 벽 세그멘테이션 ──────────────────────────────────────────────────
def get_wall_mask(img: np.ndarray, wall_model: YOLO) -> np.ndarray:
    """
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
 
 
# ── 2단계: 크랙 세그멘테이션 ────────────────────────────────────────────────
def detect_cracks_in_wall(img: np.ndarray, wall_mask: np.ndarray, crack_model: YOLO):
    """
    벽 마스크 안에 있는 크랙 후보만 반환 (분류 전 원시 결과)
    반환: (binary_masks 리스트, boxes 리스트)
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
        binary_mask  = (mask_resized > 0.5).astype(np.uint8)
 
        overlap    = np.logical_and(binary_mask, wall_mask).sum()
        crack_area = binary_mask.sum()
        if crack_area > 0 and (overlap / crack_area) >= 0.5:
            valid_masks.append(binary_mask)
            if result.boxes is not None and i < len(result.boxes):
                valid_boxes.append(result.boxes[i])
 
    return valid_masks, valid_boxes
 
 
# ── 3단계: 분류 모델로 오탐 필터링 ──────────────────────────────────────────
def classify_crack_candidates(img: np.ndarray,
                               crack_masks: list,
                               crack_boxes: list,
                               class_model: nn.Module,
                               class_device: torch.device):
    """
    각 크랙 후보 bbox 영역을 크롭해서 EfficientNet-B0 분류 모델에 통과.
    positive 판정된 것만 최종 크랙으로 인정.
 
    반환:
        accepted_masks  : positive 판정 마스크 리스트
        accepted_boxes  : positive 판정 boxes 리스트
        rejected_masks  : negative(오탐) 마스크 리스트  ← 시각화용
        classify_details: 각 후보의 분류 결과 상세 (로깅/CSV용)
    """
    h, w = img.shape[:2]
 
    # OpenCV BGR → RGB 변환 (EfficientNet은 RGB 입력)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
 
    accepted_masks   = []
    accepted_boxes   = []
    rejected_masks   = []
    classify_details = []
 
    for i, (mask, box) in enumerate(zip(crack_masks, crack_boxes)):
 
        # ── bbox 기반 크롭 ──────────────────────────────────────────────
        x1, y1, x2, y2 = map(int, box.xyxy[0])
 
        x1c = max(0, x1 - CROP_PADDING)
        y1c = max(0, y1 - CROP_PADDING)
        x2c = min(w, x2 + CROP_PADDING)
        y2c = min(h, y2 + CROP_PADDING)
 
        crop = img_rgb[y1c:y2c, x1c:x2c]
 
        if crop.size == 0:
            # 크롭 실패 → 보수적으로 positive 처리
            accepted_masks.append(mask)
            accepted_boxes.append(box)
            classify_details.append({"idx": i, "verdict": "positive(crop_fail)", "conf": -1.0})
            continue
 
        # ── EfficientNet-B0 추론 ─────────────────────────────────────────
        input_tensor = CLASSIFY_TRANSFORM(crop).unsqueeze(0).to(class_device)  # (1,3,224,224)
 
        with torch.no_grad():
            logits = class_model(input_tensor)                    # (1, 2)
            probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()  # (2,)
 
        top_idx  = int(np.argmax(probs))
        pos_conf = float(probs[POSITIVE_CLASS_INDEX])
 
        # positive 판정 조건: argmax가 positive 이고 신뢰도 >= CLASS_CONF
        is_positive = (top_idx == POSITIVE_CLASS_INDEX) and (pos_conf >= CLASS_CONF)
 
        verdict = "positive" if is_positive else "negative"
        classify_details.append({"idx": i, "verdict": verdict, "conf": pos_conf})
 
        if is_positive:
            accepted_masks.append(mask)
            accepted_boxes.append(box)
        else:
            rejected_masks.append(mask)
 
    return accepted_masks, accepted_boxes, rejected_masks, classify_details
 
 
# ── 시각화 & 저장 ────────────────────────────────────────────────────────────
def visualize_and_save(image_path: str, img: np.ndarray,
                       wall_mask: np.ndarray,
                       accepted_masks: list, accepted_boxes: list,
                       rejected_masks: list) -> str:
    """
    벽(초록) / 최종 크랙(빨강) / 오탐 제거(회색) 오버레이 후 저장
    """
    save_dir = Path(OUTPUT_DIR) / "annotated"
    save_dir.mkdir(parents=True, exist_ok=True)
 
    overlay    = img.copy()
    result_img = img.copy()
 
    # 벽 마스크 (초록)
    overlay[wall_mask == 1] = WALL_COLOR
 
    # 오탐 제거된 크랙 (회색, 반투명하게만)
    for mask in rejected_masks:
        overlay[mask == 1] = REJECT_COLOR
 
    # 최종 크랙 (빨강)
    for i, mask in enumerate(accepted_masks):
        overlay[mask == 1] = CRACK_COLOR
 
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result_img, contours, -1, CRACK_COLOR, 2)
 
        if i < len(accepted_boxes):
            conf_score = float(accepted_boxes[i].conf[0])
            x1, y1, x2, y2 = map(int, accepted_boxes[i].xyxy[0])
            cv2.putText(result_img, f"crack {conf_score:.2f}",
                        (x1, max(y1 - 8, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, CRACK_COLOR, 2)
 
    blended = cv2.addWeighted(overlay, MASK_ALPHA, result_img, 1 - MASK_ALPHA, 0)
 
    # 상태 텍스트
    wall_pct = wall_mask.sum() / wall_mask.size * 100
    cv2.putText(blended,
                f"Wall: {wall_pct:.1f}%  |  Cracks(accepted): {len(accepted_masks)}  |  Rejected: {len(rejected_masks)}",
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
 
    # 균열 비율 텍스트
    if wall_mask.sum() > 0 and accepted_masks:
        crack_total = sum(m.sum() for m in accepted_masks)
        crack_pct   = crack_total / wall_mask.sum() * 100
        color       = (0, 0, 255) if crack_pct > 5.0 else (0, 200, 0)
        cv2.putText(blended,
                    f"Crack ratio (in wall): {crack_pct:.2f}%",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
 
    # 범례
    cv2.rectangle(blended, (10,  90), (30, 110), WALL_COLOR,   -1)
    cv2.putText(blended,  "Wall",     (35, 105),  cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
    cv2.rectangle(blended, (90,  90), (110, 110), CRACK_COLOR,  -1)
    cv2.putText(blended,  "Crack",    (115, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
    cv2.rectangle(blended, (175, 90), (195, 110), REJECT_COLOR, -1)
    cv2.putText(blended,  "Rejected", (200, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
 
    stem      = Path(image_path).stem
    suffix    = Path(image_path).suffix
    save_path = save_dir / f"{stem}_result{suffix}"
    cv2.imwrite(str(save_path), blended)
    return str(save_path)
 
 
# ── 통계 계산 ────────────────────────────────────────────────────────────────
def calc_stats(wall_mask: np.ndarray,
               accepted_masks: list,
               rejected_masks: list,
               classify_details: list) -> dict:
    """이미지 1장 통계 계산"""
    wall_px   = int(wall_mask.sum())
    total_px  = int(wall_mask.size)
    wall_pct  = round(wall_px / total_px * 100, 4) if total_px > 0 else 0.0
 
    crack_px  = int(sum(m.sum() for m in accepted_masks)) if accepted_masks else 0
    crack_pct = round(crack_px / wall_px * 100, 4) if wall_px > 0 else 0.0
 
    n_rejected = len(rejected_masks)
 
    # positive 평균 신뢰도
    pos_confs = [d["conf"] for d in classify_details if d["verdict"] == "positive" and d["conf"] >= 0]
    avg_pos_conf = round(float(np.mean(pos_confs)), 4) if pos_confs else 0.0
 
    return {
        "n_cracks_raw":   len(accepted_masks) + n_rejected,  # 분류 전 후보 수
        "n_cracks":       len(accepted_masks),                # 분류 후 최종 크랙 수
        "n_rejected":     n_rejected,
        "avg_pos_conf":   avg_pos_conf,
        "wall_px":        wall_px,
        "wall_pct":       wall_pct,
        "crack_px":       crack_px,
        "crack_pct":      crack_pct,
    }
 
 
# ── CSV 저장 ─────────────────────────────────────────────────────────────────
def save_csv(rows: list, total: int, crack_detected: int):
    csv_path = Path(OUTPUT_DIR) / "summary.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
 
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
 
        writer.writerow(["실행 시각",           datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
        writer.writerow(["벽 모델",              WALL_MODEL])
        writer.writerow(["크랙 모델",            CRACK_MODEL])
        writer.writerow(["분류 모델",            CLASS_MODEL])
        writer.writerow(["분류 Positive 임계값", CLASS_CONF])
        writer.writerow(["이미지 폴더",          IMAGE_DIR])
        writer.writerow(["총 이미지",            total])
        writer.writerow(["균열 감지 이미지",     crack_detected])
        writer.writerow([])
 
        writer.writerow([
            "파일명",
            "후보 크랙 수(분류 전)",
            "최종 크랙 수(분류 후)",
            "오탐 제거 수",
            "평균 positive 신뢰도",
            "벽 면적(px)",
            "벽 비율(%)",
            "균열 면적(px)",
            "벽 대비 균열 비율(%)",
            "상태",
        ])
 
        writer.writerows(rows)
 
    print(f"\n  CSV 저장 완료: {csv_path}")
    return str(csv_path)
 
 
# ── 메인 처리 루프 ───────────────────────────────────────────────────────────
def process_folder():
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
 
    wall_model, crack_model, class_model, class_device = load_models()
 
    rows           = []
    crack_detected = 0
 
    for idx, img_path in enumerate(img_paths, 1):
        print(f"  [{idx:>4}/{total}] {img_path.name}")
 
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"           ↳ 읽기 실패, 건너뜁니다.")
            rows.append([img_path.name, "-", "-", "-", "-", "-", "-", "-", "-", "읽기 실패"])
            continue
 
        # 1단계: 벽 감지
        wall_mask = get_wall_mask(img, wall_model)
        if wall_mask.sum() == 0:
            print(f"           ↳ 벽 미감지, 크랙 감지 생략")
            rows.append([img_path.name, 0, 0, 0, 0.0, 0, 0.0, 0, 0.0, "벽 미감지"])
            continue
 
        # 2단계: 크랙 후보 감지
        crack_masks_raw, crack_boxes_raw = detect_cracks_in_wall(img, wall_mask, crack_model)
 
        # 3단계: 분류 모델로 오탐 필터링
        accepted_masks, accepted_boxes, rejected_masks, classify_details = classify_crack_candidates(
            img, crack_masks_raw, crack_boxes_raw, class_model, class_device
        )
 
        # 통계
        stats = calc_stats(wall_mask, accepted_masks, rejected_masks, classify_details)
 
        # 시각화 저장
        visualize_and_save(
            str(img_path), img, wall_mask,
            accepted_masks, accepted_boxes, rejected_masks
        )
 
        # 상태
        if stats["n_cracks"] > 0:
            crack_detected += 1
            status = f"균열 {stats['crack_pct']:.2f}%"
        else:
            status = "균열 없음"
 
        print(f"           ↳ 벽 {stats['wall_pct']:.1f}%  |  "
              f"후보 {stats['n_cracks_raw']}개 → "
              f"최종 {stats['n_cracks']}개 (제거 {stats['n_rejected']}개)  |  "
              f"벽 대비 균열 {stats['crack_pct']:.2f}%")
 
        rows.append([
            img_path.name,
            stats["n_cracks_raw"],
            stats["n_cracks"],
            stats["n_rejected"],
            stats["avg_pos_conf"],
            stats["wall_px"],
            stats["wall_pct"],
            stats["crack_px"],
            stats["crack_pct"],
            status,
        ])
 
    csv_path = save_csv(rows, total, crack_detected)
 
    print(f"\n{'─'*60}")
    print(f"  완료         : {total}장")
    print(f"  균열 감지    : {crack_detected}장  ({crack_detected/total*100:.1f}%)")
    print(f"  결과 이미지  : {Path(OUTPUT_DIR) / 'annotated'}")
    print(f"  CSV 요약     : {csv_path}")
    print(f"{'─'*60}\n")
 
 
if __name__ == "__main__":
    process_folder()