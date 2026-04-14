"""
Wall → Crack → Classification 3단계 Segmentation 파이프라인
  1단계: 벽 세그멘테이션 모델 (YOLOv11n-seg) 로 벽 인스턴스별 마스크 감지
  2단계: 전체 벽 영역으로 크랙 후보 감지
  3단계: 크랙 후보를 EfficientNet-B0 분류 모델로 오탐 필터링
  통계 : 크랙이 실제로 겹치는 벽(들)의 픽셀만 분모로 사용 → 해당 벽 대비 크랙 비율
 
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
WALL_MODEL   = r"./model_file/wall_segment_ver13.pt" # 이미지사이즈 448로 낮추고 학습 결과 좋음 
CRACK_MODEL  = r"./model_file/crack_seg_lab.pt" #ver2,ver3너무 안좋음 ver4는 ver1에서 안동펌프장 fine-tuning 후 사용
CLASS_MODEL  = r"./model_file/model.pth"  # EfficientNet-B0 state_dict
 
IMAGE_DIR    = r"./test_images"
 
WALL_CONF    = 0.25
CRACK_CONF   = 0.25
CLASS_CONF   = 0.50    # positive 판정 최소 신뢰도
IOU          = 0.45
 
# ※ ImageFolder 알파벳 순: positive=0, negative=1
POSITIVE_CLASS_INDEX = 0
 
CROP_PADDING = 20
 
WALL_COLOR   = (0, 255, 0)
CRACK_COLOR  = (0, 0, 255)
REJECT_COLOR = (128, 128, 128)
MASK_ALPHA   = 0.4
OUTPUT_DIR   = r"./results_ver19"
 
IMAGE_EXTS   = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
 
CLASSIFY_TRANSFORM = transforms.Compose([  # 학습 val_transform과 동일하게 맞춤
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
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
 
 
# ── 1단계: 벽 세그멘테이션 (인스턴스별 분리) ────────────────────────────────
def get_wall_masks(img: np.ndarray, wall_model: YOLO) -> tuple:
    """
    벽 인스턴스를 개별 마스크 리스트로 반환 (합치지 않음)
 
    반환:
        wall_masks_list : [(H,W) 이진 마스크, ...] 벽 인스턴스별
        combined_mask   : (H,W) 전체 벽 합산 마스크 (크랙 감지 입력 / 시각화용)
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
 
 
# ── 2단계: 크랙 세그멘테이션 ────────────────────────────────────────────────
def detect_cracks_in_wall(img: np.ndarray, combined_mask: np.ndarray, crack_model: YOLO):
    """
    전체 벽 합산 마스크 영역에서 크랙 후보 감지
    반환: (binary_masks 리스트, boxes 리스트)
    """
    masked_img = img.copy()
    masked_img[combined_mask == 0] = 0
 
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
 
        overlap    = np.logical_and(binary_mask, combined_mask).sum()
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
 
        top_idx  = int(np.argmax(probs))
        pos_conf = float(probs[POSITIVE_CLASS_INDEX])
        is_positive = (top_idx == POSITIVE_CLASS_INDEX) and (pos_conf >= CLASS_CONF)
 
        verdict = "positive" if is_positive else "negative"
        classify_details.append({"idx": i, "verdict": verdict, "conf": pos_conf})
 
        if is_positive:
            accepted_masks.append(mask)
            accepted_boxes.append(box)
        else:
            rejected_masks.append(mask)
 
    return accepted_masks, accepted_boxes, rejected_masks, classify_details
 
 
# ── 핵심: 크랙이 겹치는 벽만 찾아 해당 벽 픽셀 기준 비율 계산 ───────────────
def calc_crack_wall_ratio(wall_masks_list: list, accepted_masks: list) -> tuple:
    """
    크랙 픽셀과 실제로 겹치는 벽 인스턴스만 추려서
    해당 벽들의 합산 픽셀을 분모로 사용
 
    반환:
        cracked_wall_mask : 크랙이 있는 벽만 합산한 마스크 (시각화용)
        cracked_wall_px   : 크랙이 있는 벽 픽셀 합계
        crack_px          : 크랙 픽셀 합계
        crack_pct         : cracked_wall_px 대비 crack_px 비율 (%)
        n_cracked_walls   : 크랙이 감지된 벽 인스턴스 수
    """
    if not accepted_masks or not wall_masks_list:
        return np.zeros_like(wall_masks_list[0] if wall_masks_list else np.zeros((1,1), dtype=np.uint8)), 0, 0, 0.0, 0
 
    # 전체 크랙 합산 마스크
    all_crack_mask = np.zeros_like(wall_masks_list[0], dtype=np.uint8)
    for cm in accepted_masks:
        all_crack_mask = np.maximum(all_crack_mask, cm)
 
    cracked_wall_mask = np.zeros_like(wall_masks_list[0], dtype=np.uint8)
    n_cracked_walls   = 0
 
    for wm in wall_masks_list:
        # 해당 벽과 크랙이 1픽셀 이상 겹치면 "크랙 있는 벽"으로 판정
        if np.logical_and(wm, all_crack_mask).sum() > 0:
            cracked_wall_mask = np.maximum(cracked_wall_mask, wm)
            n_cracked_walls  += 1
 
    cracked_wall_px = int(cracked_wall_mask.sum())
    crack_px        = int(all_crack_mask.sum())
    crack_pct       = round(crack_px / cracked_wall_px * 100, 4) if cracked_wall_px > 0 else 0.0
 
    return cracked_wall_mask, cracked_wall_px, crack_px, crack_pct, n_cracked_walls
 
 
# ── 통계 계산 ────────────────────────────────────────────────────────────────
def calc_stats(wall_masks_list: list,
               combined_mask: np.ndarray,
               accepted_masks: list,
               rejected_masks: list,
               classify_details: list) -> dict:
 
    total_px  = int(combined_mask.size)
    wall_px   = int(combined_mask.sum())          # 전체 벽 합산
    wall_pct  = round(wall_px / total_px * 100, 4) if total_px > 0 else 0.0
 
    cracked_wall_mask, cracked_wall_px, crack_px, crack_pct, n_cracked_walls = \
        calc_crack_wall_ratio(wall_masks_list, accepted_masks)
 
    n_rejected   = len(rejected_masks)
    pos_confs    = [d["conf"] for d in classify_details if d["verdict"] == "positive" and d["conf"] >= 0]
    avg_pos_conf = round(float(np.mean(pos_confs)), 4) if pos_confs else 0.0
 
    return {
        "n_walls":         len(wall_masks_list),
        "n_cracked_walls": n_cracked_walls,        # 크랙이 감지된 벽 수
        "n_cracks_raw":    len(accepted_masks) + n_rejected,
        "n_cracks":        len(accepted_masks),
        "n_rejected":      n_rejected,
        "avg_pos_conf":    avg_pos_conf,
        "wall_px":         wall_px,                # 전체 벽 픽셀 (참고용)
        "wall_pct":        wall_pct,
        "cracked_wall_px": cracked_wall_px,        # 크랙 있는 벽만의 픽셀 (분모)
        "crack_px":        crack_px,
        "crack_pct":       crack_pct,              # 핵심 지표: 크랙 있는 벽 대비 크랙 비율
        "cracked_wall_mask": cracked_wall_mask,    # 시각화 전달용
    }
 
 
# ── 시각화 & 저장 ────────────────────────────────────────────────────────────
def visualize_and_save(image_path: str, img: np.ndarray,
                       combined_mask: np.ndarray,
                       cracked_wall_mask: np.ndarray,
                       accepted_masks: list, accepted_boxes: list,
                       rejected_masks: list,
                       stats: dict) -> str:
    save_dir = Path(OUTPUT_DIR) / "annotated"
    save_dir.mkdir(parents=True, exist_ok=True)
 
    overlay    = img.copy()
    result_img = img.copy()
 
    # 전체 벽 (연한 초록) → 크랙 없는 벽
    overlay[combined_mask == 1] = WALL_COLOR
 
    # 오탐 제거된 크랙 (회색)
    for mask in rejected_masks:
        overlay[mask == 1] = REJECT_COLOR
 
    # 최종 크랙 (빨강) + 컨투어
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
 
    # 상단 텍스트
    wall_pct = stats["wall_pct"]
    cv2.putText(blended,
                f"Walls: {stats['n_walls']}  |  Cracked walls: {stats['n_cracked_walls']}  |  "
                f"Cracks: {stats['n_cracks']}  |  Rejected: {stats['n_rejected']}",
                (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
 
    # 균열 비율 (크랙 있는 벽 기준)
    if stats["crack_pct"] > 0:
        color = (0, 0, 255) if stats["crack_pct"] > 5.0 else (0, 200, 0)
        cv2.putText(blended,
                    f"Crack ratio (cracked wall only): {stats['crack_pct']:.2f}%  "
                    f"[{stats['cracked_wall_px']}px wall / {stats['crack_px']}px crack]",
                    (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
 
    # 범례
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
            "감지된 벽 수",
            "균열 있는 벽 수",
            "후보 크랙 수(분류 전)",
            "최종 크랙 수(분류 후)",
            "오탐 제거 수",
            "평균 positive 신뢰도",
            "전체 벽 픽셀(참고)",
            "전체 벽 비율(%)(참고)",
            "균열 있는 벽 픽셀(분모)",
            "균열 픽셀",
            "균열 있는 벽 대비 균열 비율(%)",   # ← 핵심 지표
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
            rows.append([img_path.name] + ["-"] * 12)
            continue
 
        # 1단계: 벽 인스턴스별 감지
        wall_masks_list, combined_mask = get_wall_masks(img, wall_model)
 
        if not wall_masks_list:
            print(f"           ↳ 벽 미감지, 크랙 감지 생략")
            rows.append([img_path.name, 0, 0, 0, 0, 0, 0.0, 0, 0.0, 0, 0, 0.0, "벽 미감지"])
            continue
 
        # 2단계: 크랙 후보 감지 (전체 벽 합산 마스크 사용)
        crack_masks_raw, crack_boxes_raw = detect_cracks_in_wall(img, combined_mask, crack_model)
 
        # 3단계: 분류 모델 오탐 필터링
        accepted_masks, accepted_boxes, rejected_masks, classify_details = classify_crack_candidates(
            img, crack_masks_raw, crack_boxes_raw, class_model, class_device
        )
 
        # 통계 (크랙 있는 벽만 분모)
        stats = calc_stats(wall_masks_list, combined_mask, accepted_masks, rejected_masks, classify_details)
 
        # 시각화 저장
        visualize_and_save(
            str(img_path), img, combined_mask,
            stats["cracked_wall_mask"],
            accepted_masks, accepted_boxes, rejected_masks,
            stats
        )
 
        # 상태
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