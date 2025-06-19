# detection_utils.py

import cv2
import numpy as np
from ensemble_boxes import weighted_boxes_fusion


def draw_boxes(frame, boxes, scores, labels):
    h, w = frame.shape[:2]
    for b, s, l in zip(boxes, scores, labels):
        x1, y1, x2, y2 = [int(v * d) for v, d in zip(b, (w, h, w, h))]
        color = (0, 0, 255) if l == 0 else (0, 255, 255)
        name = ["fire", "smoke"][l]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"{name} {s:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )
    return frame


def draw_masks(frame, masks, color=(0, 0, 255), alpha=0.4):
    # if not masks:
    if masks is None or len(masks) == 0:
        return frame
    h, w = frame.shape[:2]
    for m in masks:
        m_np = m.cpu().numpy().astype(np.uint8)
        rm = cv2.resize(m_np, (w, h), interpolation=cv2.INTER_NEAREST)
        cm = np.zeros_like(frame)
        cm[rm > 0.5] = color
        frame = cv2.addWeighted(cm, alpha, frame, 1 - alpha, 0)
    return frame


# ─── 유틸 함수들 ────────────────────────────────────────────────────
def filter_boxes(boxes, scores, labels, min_area=0.001, max_area=0.4, conf_thr=0.3):
    out_b, out_s, out_l = [], [], []
    for b, s, l in zip(boxes, scores, labels):
        if s < conf_thr:
            continue
        w, h = b[2] - b[0], b[3] - b[1]
        area = w * h
        if min_area <= area <= max_area:
            out_b.append(b)
            out_s.append(s)
            out_l.append(int(l))
    return out_b, out_s, out_l


def ensemble_predictions(preds, iou_thr=0.5, skip_thr=0.001):
    all_b, all_s, all_l = [], [], []
    for p in preds:
        h, w = p.orig_shape
        b_list, s_list, l_list = [], [], []
        for box, score, lab in zip(
            p.boxes.xyxy.cpu().numpy(),
            p.boxes.conf.cpu().numpy(),
            p.boxes.cls.cpu().numpy(),
        ):
            x1, y1, x2, y2 = box
            b_list.append([x1 / w, y1 / h, x2 / w, y2 / h])
            s_list.append(float(score))
            l_list.append(int(lab))
        all_b.append(b_list)
        all_s.append(s_list)
        all_l.append(l_list)
    fb, fs, fl = weighted_boxes_fusion(
        all_b, all_s, all_l, iou_thr=iou_thr, skip_box_thr=skip_thr
    )
    return filter_boxes(fb, fs, fl)
