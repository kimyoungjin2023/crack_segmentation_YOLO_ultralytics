"""
Wall → Crack → Classification 3단계 Segmentation 파이프라인 (구조 개선판)

[핵심 변경사항]
  ❌ 기존: 벽 마스크로 이미지를 0으로 블랙아웃 → 크랙 모델 입력에 사용
           → 벽 모델 교체 시 크랙 모델 입력 분포가 달라져 감지 성능 저하

  ✅ 개선: 크랙 모델은 원본 이미지 전체에서 예측
           → 벽 마스크는 후처리 필터링 단계에서만 사용
           → 두 모델이 완전히 분리(Decoupled)되어 독립적으로 교체 가능

파이프라인:
  1단계: 벽 세그멘테이션 (YOLOv11n-seg) → 인스턴스별 마스크
  2단계: 크랙 감지 (원본 이미지 전체) → 후처리에서 벽 영역 50% 중첩 필터
  3단계: EfficientNet-B0 분류 모델 오탐 필터링

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
CRACK_MODEL  = r"./model_file/crack_seg_lab_ver5.pt"
CLASS_MODEL  = r"./model_file/five.pth"

IMAGE_DIR    = r"./test_images"

WALL_CONF    = 0.25
CRACK_CONF   = 0.30
CLASS_CONF   = 0.50
IOU          = 0.45

POSITIVE_CLASS_INDEX = 1

CROP_PADDING = 20

WALL_COLOR   = (0, 255, 0)
CRACK_COLOR  = (0, 0, 255)
REJECT_COLOR = (128, 128, 128)
MASK_ALPHA   = 0.4
OUTPUT_DIR   = r"./results_v3_4"

IMAGE_EXTS   = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# [변경없음] 분류 모델 transform
CLASSIFY_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
])
# ────────────────────────────────────────────────────────────────────────────


def load_class_model(path: str) -> tuple:
    """EfficientNet-B0 구조 선언 후 state_dict 로드"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    print(f"[INFO] 분류 모델 device    : {device}")
    return model, device


def load_models():
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


# ── 1단계: 벽 세그멘테이션 (변경없음) ────────────────────────────────────────
def get_wall_masks(img: np.ndarray, wall_model: YOLO) -> tuple:
    """
    벽 인스턴스를 개별 마스크 리스트로 반환

    반환:
        wall_masks_list : [(H,W) 이진 마스크, ...] 벽 인스턴스별
        combined_mask   : (H,W) 전체 벽 합산 마스크
    """
    h, w = img.shape[:2]
    results = wall_model.predict(
        source=img, conf=WALL_CONF, iou=IOU,
        retina_masks=True, verbose=False
    )
    result = results[0]

    wall_masks_list = []
    combined_mask   = np.zeros((h, w), dtype=np.uint8)

    if result.masks is None:
        return wall_masks_list, combined_mask

    for mask in result.masks.data.cpu().numpy():
        mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        binary       = (mask_resized > 0.5).astype(np.uint8)
        wall_masks_list.append(binary)
        combined_mask = np.maximum(combined_mask, binary)

    return wall_masks_list, combined_mask


# ── 2단계: 크랙 감지 (핵심 변경) ────────────────────────────────────────────
def detect_cracks_on_original(img: np.ndarray,
                               combined_mask: np.ndarray,
                               crack_model: YOLO) -> tuple:
    """
    [구조 개선] 크랙 모델을 원본 이미지 전체에서 실행 후
               벽 마스크 중첩률로 후처리 필터링

    기존: masked_img (벽 외부 = 0) 을 입력으로 사용
         → 벽 모델 바뀌면 크랙 모델 입력 분포도 바뀜

    개선: 원본 img 를 입력으로 사용
         → 크랙 모델은 항상 동일한 분포의 이미지를 받음
         → 벽 필터링은 후처리에서만 적용

    반환: (valid_masks 리스트, valid_boxes 리스트)
    """
    # ✅ 원본 이미지로 크랙 예측 (벽 마스킹 제거)
    results = crack_model.predict(
        source=img, conf=CRACK_CONF, iou=IOU,
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

        # ✅ 후처리: 벽 영역과 50% 이상 중첩되는 크랙만 유효
        crack_area = binary_mask.sum()
        if crack_area == 0:
            continue

        overlap = np.logical_and(binary_mask, combined_mask).sum()
        overlap_ratio = overlap / crack_area

        if overlap_ratio >= 0.5:
            valid_masks.append(binary_mask)
            if result.boxes is not None and i < len(result.boxes):
                valid_boxes.append(result.boxes[i])

    return valid_masks, valid_boxes


# ── 3단계: 분류 모델 오탐 필터링 (변경없음) ──────────────────────────────────
def classify_crack_candidates(img: np.ndarray,
                               crack_masks: list,
                               crack_boxes: list,
                               class_model: nn.Module,
                               class_device: torch.device):
    h, w     = img.shape[:2]
    img_rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    accepted_masks   = []
    accepted_boxes   = []
    rejected_masks   = []
    classify_details = []

    for i, (mask, box) in enumerate(zip(crack_masks, crack_boxes)):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        x1c = max(0, x1 - CROP_PADDING)
        y1c = max(0, y1 - CROP_PADDING)
        x2c = min(w, x2 + CROP_PADDING)
        y2c = min(h, y2 + CROP_PADDING)
        crop = img_rgb[y1c:y2c, x1c:x2c]

        if crop.size == 0:
            accepted_masks.append(mask)
            accepted_boxes.append(box)
            classify_details.append({"idx": i, "verdict": "positive(crop_fail)", "conf": -1.0})
            continue

        input_tensor = CLASSIFY_TRANSFORM(crop).unsqueeze(0).to(class_device)
        with torch.no_grad():
            logits = class_model(input_tensor)
            probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

        top_idx     = int(np.argmax(probs))
        pos_conf    = float(probs[POSITIVE_CLASS_INDEX])
        is_positive = (top_idx == POSITIVE_CLASS_INDEX) and (pos_conf >= CLASS_CONF)

        verdict = "positive" if is_positive else "negative"
        classify_details.append({"idx": i, "verdict": verdict, "conf": pos_conf})

        if is_positive:
            accepted_masks.append(mask)
            accepted_boxes.append(box)
        else:
            rejected_masks.append(mask)

    return accepted_masks, accepted_boxes, rejected_masks, classify_details


# ── 크랙 있는 벽 기준 비율 계산 (변경없음) ───────────────────────────────────
def calc_crack_wall_ratio(wall_masks_list: list, accepted_masks: list) -> tuple:
    if not accepted_masks or not wall_masks_list:
        empty = wall_masks_list[0] if wall_masks_list else np.zeros((1, 1), dtype=np.uint8)
        return np.zeros_like(empty), 0, 0, 0.0, 0

    all_crack_mask = np.zeros_like(wall_masks_list[0], dtype=np.uint8)
    for cm in accepted_masks:
        all_crack_mask = np.maximum(all_crack_mask, cm)

    cracked_wall_mask = np.zeros_like(wall_masks_list[0], dtype=np.uint8)
    n_cracked_walls   = 0

    for wm in wall_masks_list:
        if np.logical_and(wm, all_crack_mask).sum() > 0:
            cracked_wall_mask = np.maximum(cracked_wall_mask, wm)
            n_cracked_walls  += 1

    cracked_wall_px = int(cracked_wall_mask.sum())
    crack_px        = int(all_crack_mask.sum())
    crack_pct       = round(crack_px / cracked_wall_px * 100, 4) if cracked_wall_px > 0 else 0.0

    return cracked_wall_mask, cracked_wall_px, crack_px, crack_pct, n_cracked_walls


# ── 통계 계산 (변경없음) ──────────────────────────────────────────────────────
def calc_stats(wall_masks_list, combined_mask, accepted_masks, rejected_masks, classify_details):
    total_px = int(combined_mask.size)
    wall_px  = int(combined_mask.sum())
    wall_pct = round(wall_px / total_px * 100, 4) if total_px > 0 else 0.0

    cracked_wall_mask, cracked_wall_px, crack_px, crack_pct, n_cracked_walls = \
        calc_crack_wall_ratio(wall_masks_list, accepted_masks)

    n_rejected   = len(rejected_masks)
    pos_confs    = [d["conf"] for d in classify_details if d["verdict"] == "positive" and d["conf"] >= 0]
    avg_pos_conf = round(float(np.mean(pos_confs)), 4) if pos_confs else 0.0

    return {
        "n_walls":           len(wall_masks_list),
        "n_cracked_walls":   n_cracked_walls,
        "n_cracks_raw":      len(accepted_masks) + n_rejected,
        "n_cracks":          len(accepted_masks),
        "n_rejected":        n_rejected,
        "avg_pos_conf":      avg_pos_conf,
        "wall_px":           wall_px,
        "wall_pct":          wall_pct,
        "cracked_wall_px":   cracked_wall_px,
        "crack_px":          crack_px,
        "crack_pct":         crack_pct,
        "cracked_wall_mask": cracked_wall_mask,
    }


# ── 시각화 & 저장 (변경없음) ──────────────────────────────────────────────────
def visualize_and_save(image_path, img, combined_mask, cracked_wall_mask,
                       accepted_masks, accepted_boxes, rejected_masks, stats):
    save_dir = Path(OUTPUT_DIR) / "annotated"
    save_dir.mkdir(parents=True, exist_ok=True)

    overlay    = img.copy()
    result_img = img.copy()

    overlay[combined_mask == 1] = WALL_COLOR

    for mask in rejected_masks:
        overlay[mask == 1] = REJECT_COLOR

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

    cv2.putText(blended,
                f"Walls: {stats['n_walls']}  |  Cracked walls: {stats['n_cracked_walls']}  |  "
                f"Cracks: {stats['n_cracks']}  |  Rejected: {stats['n_rejected']}",
                (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if stats["crack_pct"] > 0:
        color = (0, 0, 255) if stats["crack_pct"] > 5.0 else (0, 200, 0)
        cv2.putText(blended,
                    f"Crack ratio (cracked wall only): {stats['crack_pct']:.2f}%  "
                    f"[{stats['cracked_wall_px']}px wall / {stats['crack_px']}px crack]",
                    (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    cv2.rectangle(blended, (10,  85), (30, 105), WALL_COLOR,   -1)
    cv2.putText(blended,  "Wall",     (35, 100),  cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255,255,255), 2)
    cv2.rectangle(blended, (85,  85), (105, 105), CRACK_COLOR,  -1)
    cv2.putText(blended,  "Crack",    (110, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255,255,255), 2)
    cv2.rectangle(blended, (165, 85), (185, 105), REJECT_COLOR, -1)
    cv2.putText(blended,  "Rejected", (190, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255,255,255), 2)

    stem      = Path(image_path).stem
    suffix    = Path(image_path).suffix
    save_path = save_dir / f"{stem}_result{suffix}"
    cv2.imwrite(str(save_path), blended)
    return str(save_path)


# ── CSV 저장 (변경없음) ───────────────────────────────────────────────────────
def save_csv(rows, total, crack_detected):
    csv_path = Path(OUTPUT_DIR) / "summary.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["실행 시각",           datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
        writer.writerow(["벽 모델",              WALL_MODEL])
        writer.writerow(["크랙 모델",            CRACK_MODEL])
        writer.writerow(["분류 모델",            CLASS_MODEL])
        writer.writerow(["분류 Positive 임계값", CLASS_CONF])
        writer.writerow(["파이프라인 구조",      "Decoupled (크랙 모델 = 원본 이미지 입력)"])
        writer.writerow(["이미지 폴더",          IMAGE_DIR])
        writer.writerow(["총 이미지",            total])
        writer.writerow(["균열 감지 이미지",     crack_detected])
        writer.writerow([])
        writer.writerow([
            "파일명", "감지된 벽 수", "균열 있는 벽 수",
            "후보 크랙 수(분류 전)", "최종 크랙 수(분류 후)", "오탐 제거 수",
            "평균 positive 신뢰도", "전체 벽 픽셀(참고)", "전체 벽 비율(%)(참고)",
            "균열 있는 벽 픽셀(분모)", "균열 픽셀", "균열 있는 벽 대비 균열 비율(%)", "상태",
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
    print(f"[시작] 총 {total}장 처리 시작  (파이프라인: Decoupled)")
    print(f"       결과 저장: {Path(OUTPUT_DIR).absolute()}\n")

    wall_model, crack_model, class_model, class_device = load_models()

    rows           = []
    crack_detected = 0

    for idx, img_path in enumerate(img_paths, 1):
        print(f"  [{idx:>4}/{total}] {img_path.name}")

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"           ↳ 읽기 실패, 건너뜁니다.")
            rows.append([img_path.name] + ["-"] * 12)
            continue

        # 1단계: 벽 인스턴스별 감지
        wall_masks_list, combined_mask = get_wall_masks(img, wall_model)

        if not wall_masks_list:
            print(f"           ↳ 벽 미감지, 크랙 감지 생략")
            rows.append([img_path.name, 0, 0, 0, 0, 0, 0.0, 0, 0.0, 0, 0, 0.0, "벽 미감지"])
            continue

        # 2단계: ✅ 원본 이미지로 크랙 감지 → 후처리에서 벽 필터
        crack_masks_raw, crack_boxes_raw = detect_cracks_on_original(
            img, combined_mask, crack_model
        )

        # 3단계: 분류 모델 오탐 필터링
        accepted_masks, accepted_boxes, rejected_masks, classify_details = \
            classify_crack_candidates(img, crack_masks_raw, crack_boxes_raw,
                                      class_model, class_device)

        # 통계 계산
        stats = calc_stats(wall_masks_list, combined_mask,
                           accepted_masks, rejected_masks, classify_details)

        # 시각화 저장
        visualize_and_save(
            str(img_path), img, combined_mask,
            stats["cracked_wall_mask"],
            accepted_masks, accepted_boxes, rejected_masks,
            stats
        )

        if stats["n_cracks"] > 0:
            crack_detected += 1
            status = f"균열 {stats['crack_pct']:.2f}%"
        else:
            status = "균열 없음"

        print(f"           ↳ 벽 {stats['n_walls']}개 (균열벽 {stats['n_cracked_walls']}개)  |  "
              f"후보 {stats['n_cracks_raw']}개 → 최종 {stats['n_cracks']}개 (제거 {stats['n_rejected']}개)  |  "
              f"균열벽 대비 크랙 {stats['crack_pct']:.2f}%")

        rows.append([
            img_path.name,
            stats["n_walls"],
            stats["n_cracked_walls"],
            stats["n_cracks_raw"],
            stats["n_cracks"],
            stats["n_rejected"],
            stats["avg_pos_conf"],
            stats["wall_px"],
            stats["wall_pct"],
            stats["cracked_wall_px"],
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