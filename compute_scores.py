#!/usr/bin/env python3
"""
compute_scores.py (fixed)
- resizes predicted mask to GT mask size if shapes differ (nearest neighbor)
- computes per-image iou_score, anchor_score, boolean_score
- writes metrics_per_image.csv and prints means and writes a summary row
"""
import os, glob, csv
import numpy as np
import cv2
import pandas as pd
from shapely.geometry import Point, Polygon

PRED_MASK_DIR = "predictions/masks"
GT_MASK_DIR   = "masks/test"
PRED_ANCHORS  = "predictions/anchors_pred.csv"  # filename,x1,y1,x2,y2 (normalized)
OUT_CSV = "metrics_per_image.csv"
eps = 1e-7

def read_pred_anchors(csv_path):
    anchors = {}
    if not os.path.exists(csv_path):
        print("Predicted anchors CSV not found:", csv_path)
        return anchors
    with open(csv_path, newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            fn = row['filename']
            try:
                x1 = float(row['x1']); y1 = float(row['y1'])
                x2 = float(row['x2']); y2 = float(row['y2'])
                anchors[fn] = [x1,y1,x2,y2]
            except Exception as e:
                print("Skipping bad row", row, e)
    return anchors

def iou_score(pred_mask, gt_mask):
    # both masks must be same shape (binary 0/255)
    pred = (pred_mask>127).astype(np.uint8)
    gt   = (gt_mask>127).astype(np.uint8)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0 if inter==0 else 0.0
    return float(inter) / (union + eps)

def polygon_mask_from_anchors(anchors_norm, H, W):
    if anchors_norm is None:
        return np.zeros((H,W), dtype=np.uint8), None, None
    pts = np.array(anchors_norm, dtype=float).reshape(-1,2)
    if pts.max() <= 1.01:
        pts[:,0] = pts[:,0] * W
        pts[:,1] = pts[:,1] * H
    pts_int = pts.astype(np.int32)
    poly_mask = np.zeros((H,W), dtype=np.uint8)
    try:
        cv2.fillPoly(poly_mask, [pts_int], 1)
    except Exception as e:
        return np.zeros((H,W), dtype=np.uint8), None, None
    try:
        poly_sh = Polygon([(float(x), float(y)) for x,y in pts])
    except Exception:
        poly_sh = None
    return poly_mask, poly_sh, pts_int

def anchor_score_from_seg_and_poly(pred_mask, poly_mask):
    predb = (pred_mask>127).astype(np.uint8)
    total_pred = predb.sum()
    if total_pred == 0:
        return 0.0
    inside = np.logical_and(predb, poly_mask).sum()
    return float(inside) / (total_pred + eps)

def boolean_score_from_gt_centroid(gt_mask, poly_sh):
    if poly_sh is None:
        return 0
    M = cv2.moments((gt_mask>127).astype(np.uint8))
    if M['m00'] == 0:
        H,W = gt_mask.shape
        cx,cy = W/2.0, H/2.0
    else:
        cx = M['m10']/M['m00']
        cy = M['m01']/M['m00']
    pt = Point(float(cx), float(cy))
    return 1 if (poly_sh.contains(pt) or poly_sh.touches(pt)) else 0

def main():
    pred_anchors = read_pred_anchors(PRED_ANCHORS)
    pred_mask_files = sorted(glob.glob(os.path.join(PRED_MASK_DIR, "*.png")) + glob.glob(os.path.join(PRED_MASK_DIR, "*.jpg")))
    rows = []
    missing = 0
    skipped = 0
    for pred_path in pred_mask_files:
        fname = os.path.basename(pred_path)
        gt_path = os.path.join(GT_MASK_DIR, fname)
        if not os.path.exists(gt_path):
            missing += 1
            continue
        pred_mask = cv2.imread(pred_path, 0)
        gt_mask = cv2.imread(gt_path, 0)
        if pred_mask is None or gt_mask is None:
            skipped += 1
            continue

        # If shapes differ, resize predicted mask to GT mask size (nearest)
        if pred_mask.shape != gt_mask.shape:
            H_gt, W_gt = gt_mask.shape
            pred_mask_rs = cv2.resize(pred_mask, (W_gt, H_gt), interpolation=cv2.INTER_NEAREST)
        else:
            pred_mask_rs = pred_mask

        # IOU
        iou = iou_score(pred_mask_rs, gt_mask)

        # anchors -> polygon in GT pixel space (scale normalized anchors to GT size)
        anchors = pred_anchors.get(fname, None)
        poly_mask, poly_sh, poly_pts = polygon_mask_from_anchors(anchors, gt_mask.shape[0], gt_mask.shape[1])

        anchor_s = anchor_score_from_seg_and_poly(pred_mask_rs, poly_mask)
        bool_s = boolean_score_from_gt_centroid(gt_mask, poly_sh)
        rows.append([fname, float(iou), float(anchor_s), int(bool_s)])

    if missing>0:
        print(f"Warning: {missing} predicted masks had no GT mask and were skipped.")
    if skipped>0:
        print(f"Warning: {skipped} image(s) could not be read and were skipped.")

    # write CSV with per-image rows and a summary row at the end
    header = ['ImageName','iou_score','anchor_score','boolean_score']
    df = pd.DataFrame(rows, columns=header)
    if not df.empty:
        mean_iou = df['iou_score'].mean()
        mean_anchor = df['anchor_score'].mean()
        mean_bool = df['boolean_score'].mean()
    else:
        mean_iou = mean_anchor = mean_bool = 0.0

    # append summary row
    summary_row = ['MEAN', mean_iou, mean_anchor, int(round(mean_bool))]
    df_out = df.copy()
    df_out.loc[len(df_out)] = summary_row

    df_out.to_csv(OUT_CSV, index=False)
    print("Wrote", OUT_CSV, "rows=", len(df))
    print(f"Mean IOU (iou_score): {mean_iou:.4f}")
    print(f"Mean Anchor_Score (anchor_score): {mean_anchor:.4f}")
    print(f"Mean Boolean_score (boolean_score): {mean_bool:.4f}")

if __name__=='__main__':
    main()
