"""
파이프라인 진단 스크립트
각 단계 후 중간 결과를 이미지로 저장해서 어디서 크랙이 사라지는지 확인
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from pathlib import Path
from ultralytics import YOLO

# ── 설정 (메인 코드와 동일하게) ──────────────────────────────────────────────
WALL_MODEL  = r"./model_file/wall_segment_ver13.pt"
CRACK_MODEL = r"./model_file/images_5335_segment.pt"
CLASS_MODEL = r"./model_file/five.pth"

TEST_IMAGE  = r"./test_images/Gemini_Generated_Image_q4qfazq4qfazq4qf.png"  # ← 크랙이 안나오는 이미지 경로

WALL_CONF   = 0.50
CRACK_CONF  = 0.50   # 낮춰서 후보를 최대한 많이 보기
IOU         = 0.45
CLASS_CONF  = 0.30
OVERLAP_THR = 0.50   # 벽 중첩 비율 기준
POSITIVE_CLASS_INDEX = 1

DIAG_DIR    = r"./ver2_0.50_0.50_0.3"
# ─────────────────────────────────────────────────────────────────────────────

CLASSIFY_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
])


def save(path, img, label):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)
    print(f"  💾 저장: {path}  ({label})")


def draw_masks(img, masks, color, alpha=0.45):
    overlay = img.copy()
    for m in masks:
        overlay[m == 1] = color
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)


def main():
    img = cv2.imread(TEST_IMAGE)
    assert img is not None, f"이미지를 읽을 수 없음: {TEST_IMAGE}"
    h, w = img.shape[:2]
    name = Path(TEST_IMAGE).stem
    base = Path(DIAG_DIR) / name

    print(f"\n{'='*60}")
    print(f"  진단 대상 : {TEST_IMAGE}  ({w}x{h})")
    print(f"{'='*60}\n")

    # ── STEP 1: 벽 감지 ───────────────────────────────────────────────────
    print("▶ STEP 1 — 벽 세그멘테이션")
    wall_model = YOLO(WALL_MODEL)
    res = wall_model.predict(source=img, conf=WALL_CONF, iou=IOU,
                              retina_masks=True, verbose=False)[0]

    wall_masks_list = []
    combined_mask   = np.zeros((h, w), dtype=np.uint8)

    if res.masks is not None:
        for mask in res.masks.data.cpu().numpy():
            mr = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            b  = (mr > 0.5).astype(np.uint8)
            wall_masks_list.append(b)
            combined_mask = np.maximum(combined_mask, b)

    print(f"   감지된 벽 수 : {len(wall_masks_list)}")
    print(f"   벽 픽셀 합계 : {combined_mask.sum()}")

    vis1 = draw_masks(img, wall_masks_list, (0, 255, 0))
    cv2.putText(vis1, f"STEP1 walls={len(wall_masks_list)}", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    save(base / "step1_wall.jpg", vis1, "벽 감지 결과")

    if not wall_masks_list:
        print("   ❌ 벽이 감지되지 않았습니다. 여기서 중단.")
        return

    # ── STEP 2A: 크랙 후보 — 원본 이미지 (개선판) ────────────────────────
    print("\n▶ STEP 2A — 크랙 감지 (원본 이미지 입력)")
    crack_model = YOLO(CRACK_MODEL)

    # 원본으로 예측
    res_orig = crack_model.predict(source=img, conf=CRACK_CONF, iou=IOU,
                                    retina_masks=True, verbose=False)[0]
    all_masks_orig = []
    if res_orig.masks is not None:
        for mask in res_orig.masks.data.cpu().numpy():
            mr = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            all_masks_orig.append((mr > 0.5).astype(np.uint8))

    print(f"   원본 입력 크랙 후보 수 : {len(all_masks_orig)}")
    vis2a = draw_masks(img, all_masks_orig, (0, 0, 255))
    cv2.putText(vis2a, f"STEP2A (original) candidates={len(all_masks_orig)}", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    save(base / "step2a_crack_original_input.jpg", vis2a, "크랙 후보 (원본 입력)")

    # ── STEP 2B: 크랙 후보 — 마스킹 이미지 (기존 방식) 비교용 ────────────
    print("\n▶ STEP 2B — 크랙 감지 (마스킹 이미지 입력, 기존 방식 비교)")
    masked_img = img.copy()
    masked_img[combined_mask == 0] = 0

    res_masked = crack_model.predict(source=masked_img, conf=CRACK_CONF, iou=IOU,
                                      retina_masks=True, verbose=False)[0]
    all_masks_masked = []
    if res_masked.masks is not None:
        for mask in res_masked.masks.data.cpu().numpy():
            mr = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            all_masks_masked.append((mr > 0.5).astype(np.uint8))

    print(f"   마스킹 입력 크랙 후보 수 : {len(all_masks_masked)}")
    vis2b = draw_masks(masked_img, all_masks_masked, (0, 0, 255))
    cv2.putText(vis2b, f"STEP2B (masked) candidates={len(all_masks_masked)}", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    save(base / "step2b_crack_masked_input.jpg", vis2b, "크랙 후보 (마스킹 입력)")

    # 마스킹 이미지 자체도 저장 (벽 모델이 어떻게 잘랐는지 확인)
    save(base / "step2b_masked_img.jpg", masked_img, "크랙 모델에 들어간 마스킹 이미지")

    # ── STEP 3: 벽 중첩 필터 ────────────────────────────────────────────
    print("\n▶ STEP 3 — 벽 중첩 필터 (원본 기준 후보에 적용)")
    passed_wall, failed_wall = [], []
    for m in all_masks_orig:
        crack_area = m.sum()
        if crack_area == 0:
            continue
        overlap_ratio = np.logical_and(m, combined_mask).sum() / crack_area
        if overlap_ratio >= OVERLAP_THR:
            passed_wall.append(m)
        else:
            failed_wall.append((m, overlap_ratio))

    print(f"   통과 : {len(passed_wall)}개")
    print(f"   탈락 : {len(failed_wall)}개")
    for m, r in failed_wall:
        print(f"          → 중첩률 {r:.3f} < {OVERLAP_THR} 로 제거됨")

    vis3_pass = draw_masks(img, passed_wall, (0, 255, 0))
    vis3_fail = draw_masks(vis3_pass, [m for m, _ in failed_wall], (128, 128, 128))
    cv2.putText(vis3_fail, f"STEP3 passed={len(passed_wall)} failed={len(failed_wall)}", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    save(base / "step3_wall_overlap_filter.jpg", vis3_fail, "벽 중첩 필터 결과")

    if not passed_wall:
        print("   ❌ 벽 중첩 필터에서 모두 탈락. CRACK_CONF 또는 OVERLAP_THR 조정 필요.")

    # ── STEP 4: 분류 모델 필터 ──────────────────────────────────────────
    if passed_wall and res_orig.boxes is not None:
        print("\n▶ STEP 4 — 분류 모델 필터 (EfficientNet-B0)")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls_model = models.efficientnet_b0(weights=None)
        cls_model.classifier[1] = nn.Linear(cls_model.classifier[1].in_features, 2)
        cls_model.load_state_dict(torch.load(CLASS_MODEL, map_location=device))
        cls_model.to(device).eval()

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        passed_cls, failed_cls = [], []

        # passed_wall 인덱스와 boxes 인덱스 매핑
        valid_indices = []
        if res_orig.masks is not None:
            for i, mask in enumerate(res_orig.masks.data.cpu().numpy()):
                mr = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                b  = (mr > 0.5).astype(np.uint8)
                crack_area = b.sum()
                if crack_area == 0:
                    continue
                overlap_ratio = np.logical_and(b, combined_mask).sum() / crack_area
                if overlap_ratio >= OVERLAP_THR:
                    valid_indices.append(i)

        for rank, (m, orig_i) in enumerate(zip(passed_wall, valid_indices)):
            if orig_i >= len(res_orig.boxes):
                continue
            box = res_orig.boxes[orig_i]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = img_rgb[max(0,y1-20):min(h,y2+20), max(0,x1-20):min(w,x2+20)]
            if crop.size == 0:
                passed_cls.append(m)
                continue

            t = CLASSIFY_TRANSFORM(crop).unsqueeze(0).to(device)
            with torch.no_grad():
                probs = torch.softmax(cls_model(t), dim=1).squeeze().cpu().numpy()
            pos_conf = float(probs[POSITIVE_CLASS_INDEX])
            top_idx  = int(np.argmax(probs))
            is_pos   = (top_idx == POSITIVE_CLASS_INDEX) and (pos_conf >= CLASS_CONF)

            print(f"   후보 {rank}: positive_conf={pos_conf:.4f}  "
                  f"top_class={top_idx}  → {'✅ positive' if is_pos else '❌ negative (rejected)'}")

            if is_pos:
                passed_cls.append(m)
            else:
                failed_cls.append(m)

        print(f"\n   통과 : {len(passed_cls)}개")
        print(f"   탈락 : {len(failed_cls)}개")

        vis4 = draw_masks(img, passed_cls, (0, 0, 255))
        vis4 = draw_masks(vis4, failed_cls, (128, 128, 128))
        cv2.putText(vis4, f"STEP4 passed={len(passed_cls)} rejected={len(failed_cls)}", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        save(base / "step4_classification_filter.jpg", vis4, "분류 필터 결과")

    # ── 진단 요약 ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  진단 요약")
    print(f"{'='*60}")
    print(f"  STEP1 벽 감지          : {len(wall_masks_list)}개")
    print(f"  STEP2A 크랙 후보 (원본): {len(all_masks_orig)}개")
    print(f"  STEP2B 크랙 후보 (마스): {len(all_masks_masked)}개  ← 기존 방식")
    print(f"  STEP3  벽 중첩 통과    : {len(passed_wall)}개")
    print(f"  저장 위치              : {Path(DIAG_DIR) / name}")
    print(f"{'='*60}\n")

    # 임계값 조정 힌트
    if len(all_masks_orig) == 0:
        print("  💡 조치: CRACK_CONF를 낮춰보세요 (현재 0.25 → 0.1 시도)")
        print("           또는 크랙 모델이 원본 이미지에 미적응 상태일 수 있음")
        print("           → STEP2B(마스킹 입력)와 비교해서 어느쪽이 많이 나오는지 확인")
    elif len(passed_wall) == 0:
        print("  💡 조치: OVERLAP_THR를 낮춰보세요 (현재 0.50 → 0.30 시도)")
    else:
        print("  💡 조치: CLASS_CONF를 낮춰보세요 (현재 0.50 → 0.30 시도)")
        print("           또는 five.pth 분류 모델이 과도하게 reject 중")


if __name__ == "__main__":
    main()