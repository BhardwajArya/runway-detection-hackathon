#!/usr/bin/env python3
"""
inference.py
Run trained runway model on test set:
- Loads best_runway.pth
- Predicts segmentation masks and anchors
- Saves masks in predictions/masks/
- Saves anchors in predictions/anchors_pred.csv
- Builds submission.csv
"""

import os, csv, glob
import torch
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from deep_runway import RunwayModel, get_transforms

# config
INPUT_H, INPUT_W = 256, 448
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = 'best_runway.pth'
IMAGES_DIR = 'images/test'
OUT_MASK_DIR = 'predictions/masks'
ANCHOR_CSV = 'predictions/anchors_pred.csv'
SUBMISSION_CSV = 'submission.csv'

os.makedirs(OUT_MASK_DIR, exist_ok=True)

# load model
model = RunwayModel(encoder_name='resnet34')
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval().to(DEVICE)

# preprocessing
transform = get_transforms(train=False, size=(INPUT_H, INPUT_W))

rows = []
test_images = sorted(glob.glob(os.path.join(IMAGES_DIR, "*.png")) + glob.glob(os.path.join(IMAGES_DIR, "*.jpg")))

for img_path in tqdm(test_images):
    fname = os.path.basename(img_path)

    img = cv2.imread(img_path)[:,:,::-1]
    aug = transform(image=img)
    img_t = np.transpose(aug['image'].astype('float32')/255.0, (2,0,1))
    img_t = torch.tensor(img_t).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        seg_pred, kp_pred = model(img_t)
        seg_np = (seg_pred.squeeze().cpu().numpy() > 0.5).astype('uint8')*255
        kp_np = kp_pred.squeeze().cpu().numpy()

    # save mask
    out_mask_path = os.path.join(OUT_MASK_DIR, os.path.splitext(fname)[0] + ".png")
    cv2.imwrite(out_mask_path, seg_np)

    # save anchors row
    rows.append([fname] + kp_np.tolist())

# write anchors csv
with open(ANCHOR_CSV, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['filename','x1','y1','x2','y2'])
    for r in rows:
        w.writerow(r)

# build submission.csv (same format)
pd.DataFrame(rows, columns=['filename','x1','y1','x2','y2']).to_csv(SUBMISSION_CSV, index=False)

print(f"✅ Inference complete. Masks -> {OUT_MASK_DIR}, Anchors -> {ANCHOR_CSV}, Submission -> {SUBMISSION_CSV}")
