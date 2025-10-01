#!/usr/bin/env python3
"""
json_to_masks.py  -- simple converter for FS2020 labels/lines JSON structure.

Assumes JSON is a dict: filename -> list of { "label": "LEDG"/"REDG"/"CTL"/..., "points": [[x1,y1],[x2,y2]] ... }

It creates a mask per image by forming a polygon from LEDG and REDG lines:
  polygon = [LEDG.p0, LEDG.p1, REDG.p1, REDG.p0]

Usage example (run from runway_project):
python json_to_masks.py --json /full/path/to/train_labels_640x360.json \
  --images_dir images/train --out_dir masks/train --width 640 --height 360
"""
import os, json, argparse, cv2, numpy as np

def load_json(p):
    with open(p,'r') as f:
        return json.load(f)

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def get_lines_for_image(entry_list):
    # returns dict label -> points (list of 2 points) for the entry
    d = {}
    for it in entry_list:
        lbl = it.get('label')
        pts = it.get('points') or it.get('points_list') or it.get('polygon') or None
        if not pts:
            # sometimes points are nested under 'shape' etc
            for k in ('points','pts','shape','poly'):
                if k in it:
                    pts = it[k]; break
        if lbl and pts and len(pts) >= 2:
            d[lbl] = pts
    return d

def make_mask_for_fname(fname, lines_map, images_dir, out_dir, W, H):
    # lines_map: dict mapping filename -> list-of-entries
    entries = lines_map.get(fname, [])
    # entries here may be the list of dicts we saw
    label_map = get_lines_for_image(entries) if isinstance(entries, list) else {}
    # pick LEDG and REDG
    led = label_map.get('LEDG') or label_map.get('ledg') or label_map.get('LeftEdge')
    red = label_map.get('REDG') or label_map.get('redg') or label_map.get('RightEdge')
    # if missing, try to fallback using any two longest lines
    if not led or not red:
        # try to find any two line entries
        if isinstance(entries, list) and len(entries) >= 2:
            # sort by length and pick two longest as left/right
            def length(e):
                pts = e.get('points') if isinstance(e, dict) else None
                if not pts: return 0
                a,b = np.array(pts[0]), np.array(pts[1])
                return np.linalg.norm(a-b)
            sorted_entries = sorted(entries, key=length, reverse=True)
            if len(sorted_entries) >= 2:
                led = sorted_entries[0].get('points')
                red = sorted_entries[1].get('points')
    # build polygon if both available
    if not led or not red:
        return False, "missing LEDG/REDG"
    # led and red are each lists of two [x,y]
    try:
        l0 = (float(led[0][0]), float(led[0][1]))
        l1 = (float(led[1][0]), float(led[1][1]))
        r0 = (float(red[0][0]), float(red[0][1]))
        r1 = (float(red[1][0]), float(red[1][1]))
    except Exception as e:
        return False, f"bad pts: {e}"
    poly = np.array([l0, l1, r1, r0], dtype=np.int32)
    # determine image size
    imgpath = os.path.join(images_dir, fname)
    if os.path.exists(imgpath):
        im = cv2.imread(imgpath)
        H_img, W_img = im.shape[:2]
    else:
        W_img, H_img = W, H
    mask = np.zeros((H_img, W_img), dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    out_name = os.path.splitext(os.path.basename(fname))[0] + '.png'
    out_path = os.path.join(out_dir, out_name)
    cv2.imwrite(out_path, mask)
    return True, out_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', required=True)
    parser.add_argument('--images_dir', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=360)
    args = parser.parse_args()

    data = load_json(args.json)
    ensure_dir(args.out_dir)
    # data is mapping filename -> list-of-dicts (we saw this)
    count = 0
    failed = 0
    for i, (fname, entry_list) in enumerate(data.items()):
        ok, msg = make_mask_for_fname(fname, data, args.images_dir, args.out_dir, args.width, args.height)
        if ok:
            count += 1
        else:
            failed += 1
            if failed <= 10:
                print("FAILED", fname, msg)
        if i % 200 == 0:
            print(f"[{i}] processed {count} masks, {failed} failures")
    print("Done. wrote", count, "masks; failures:", failed)

if __name__ == '__main__':
    main()
