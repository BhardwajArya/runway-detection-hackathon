# cv_baseline.py
import os, argparse, cv2, numpy as np, csv, glob

def process_image(img_path, out_mask_path):
    img = cv2.imread(img_path)
    H,W = img.shape[:2]
    # resize for speed
    small = cv2.resize(img, (576,320))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)
    # morphological closing to fill runway area
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (15,15))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, ker)
    # dilate to merge lines
    dil = cv2.dilate(closed, np.ones((9,9), np.uint8), iterations=2)
    # threshold and find largest contour
    _,th = cv2.threshold(dil, 10, 255, cv2.THRESH_BINARY)
    cnts,_ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(th.shape, dtype=np.uint8)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        cv2.drawContours(mask, [c], -1, 255, -1)
    # upscale mask to original image size
    mask_full = cv2.resize(mask, (W,H), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(out_mask_path, mask_full)
    # compute normalized anchors from mask
    anchors = mask_to_anchors(mask_full)
    return anchors

def mask_to_anchors(mask):
    # returns normalized two anchors [x1,y1,x2,y2] in 0..1
    H,W = mask.shape
    cnts,_ = cv2.findContours((mask>127).astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return [0.5, 0.9, 0.5, 0.1]
    c = max(cnts, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    # select farthest pair
    dmax=-1; a=None; b=None
    for i in range(4):
        for j in range(i+1,4):
            d = np.sum((box[i]-box[j])**2)
            if d>dmax:
                dmax=d; a=box[i]; b=box[j]
    ax,ay = a[0]/W, a[1]/H
    bx,by = b[0]/W, b[1]/H
    return [ax,ay,bx,by]

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', default='images/test', help='input images dir')
    p.add_argument('--out_masks', default='predictions/masks', help='output masks dir')
    p.add_argument('--out_anchors', default='predictions/anchors_pred.csv', help='anchors CSV')
    args = p.parse_args()
    os.makedirs(args.out_masks, exist_ok=True)
    rows=[]
    for img_path in sorted(glob.glob(os.path.join(args.input,'*.png')) + glob.glob(os.path.join(args.input,'*.jpg'))):
        name = os.path.basename(img_path)
        out_mask = os.path.join(args.out_masks, name)
        anchors = process_image(img_path, out_mask)
        rows.append([name] + anchors)
        print('done', name)
    with open(args.out_anchors, 'w', newline='') as f:
        import csv
        w = csv.writer(f)
        w.writerow(['filename','x1','y1','x2','y2'])
        w.writerows(rows)
    print('Saved masks ->', args.out_masks, 'anchors ->', args.out_anchors)
