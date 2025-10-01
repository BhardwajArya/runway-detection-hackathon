import os, cv2, csv, glob
out='anchors_gt.csv'
masks=sorted(glob.glob('masks/**/*.png', recursive=True))
rows=[]
for m in masks:
    name=os.path.basename(m)
    mask=cv2.imread(m,0)
    if mask is None:
        continue
    contours,_=cv2.findContours((mask>127).astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        rows.append([name,0.5,0.9,0.5,0.1]); continue
    c=max(contours,key=cv2.contourArea)
    rect=cv2.minAreaRect(c)
    box=cv2.boxPoints(rect)
    # choose two farthest points
    a,b=None,None; dmax=-1
    for i in range(4):
        for j in range(i+1,4):
            d=((box[i][0]-box[j][0])**2+(box[i][1]-box[j][1])**2)
            if d> dmax:
                dmax=d; a=box[i]; b=box[j]
    H,W = mask.shape
    rows.append([name, float(a[0]/W), float(a[1]/H), float(b[0]/W), float(b[1]/H)])
# write CSV
with open(out,'w',newline='') as f:
    w=csv.writer(f); w.writerow(['filename','x1','y1','x2','y2']); w.writerows(rows)
print("Wrote", out, "rows=", len(rows))

