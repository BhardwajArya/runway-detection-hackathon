import os, argparse, csv, math, random
from glob import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import albumentations as A
import segmentation_models_pytorch as smp

# -----------------------
# Dataset
# -----------------------
class RunwayDataset(Dataset):
    def __init__(self, csv_file, images_dir='images/train', masks_dir='masks/train', anchors_csv=None, transform=None, input_size=(320,576)):
        self.df = pd.read_csv(csv_file)
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.h, self.w = input_size
        # load anchors if provided
        self.anchors = {}
        if anchors_csv and os.path.exists(anchors_csv):
            adf = pd.read_csv(anchors_csv)
            for _, r in adf.iterrows():
                self.anchors[str(r['filename'])] = [r['x1'], r['y1'], r['x2'], r['y2']]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        fname = str(self.df.iloc[idx][0])
        img_path = None
        # try images_dir/fname
        cand = os.path.join(self.images_dir, fname)
        if os.path.exists(cand):
            img_path = cand
        else:
            # fallback: try match basename in images_dir
            base = os.path.basename(fname)
            files = glob(os.path.join(self.images_dir, base))
            if files:
                img_path = files[0]
        if img_path is None:
            raise FileNotFoundError(f"Image not found: {fname} in {self.images_dir}")

        img = cv2.imread(img_path)[:,:,::-1]
        # load mask
        mask_path = os.path.join(self.masks_dir, os.path.splitext(os.path.basename(fname))[0] + '.png')
        if not os.path.exists(mask_path):
            # fallback: find any matching mask by basename
            cand = glob(os.path.join(self.masks_dir, os.path.basename(fname)))
            if cand: mask_path = cand[0]
        mask = cv2.imread(mask_path, 0) if os.path.exists(mask_path) else np.zeros((img.shape[0], img.shape[1]), np.uint8)
        mask = (mask > 127).astype('uint8')

        # augmentation / resize
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']; mask = augmented['mask']
        else:
            img = cv2.resize(img, (self.w, self.h))
            mask = cv2.resize(mask, (self.w, self.h), interpolation=cv2.INTER_NEAREST)

        img = img.astype('float32') / 255.0
        img = np.transpose(img, (2,0,1))
        mask = mask.astype('float32')[None,:,:]

        # anchors
        anchors = self.anchors.get(os.path.basename(fname), [-1,-1,-1,-1])
        anchors = np.array(anchors, dtype=np.float32)

        return torch.tensor(img), torch.tensor(mask), torch.tensor(anchors), fname

# -----------------------
# Augmentations
# -----------------------
def get_transforms(train=True, size=(320,576)):
    h,w = size
    if train:
        return A.Compose([
            A.Resize(h,w),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=12, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.GaussNoise(p=0.2),
        ])
    else:
        return A.Compose([A.Resize(h,w)])

# -----------------------
# Model (UNet encoder + small kp head)
# -----------------------
class RunwayModel(nn.Module):
    def __init__(self, encoder_name='resnet34', encoder_weights='imagenet'):
        super().__init__()
        # segmentation model from SMP
        self.seg = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1
        )
        # regression head on deepest encoder features
        enc_ch = self.seg.encoder.out_channels[-1]
        self.kp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(enc_ch, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 4),
            nn.Sigmoid()
        )

    def forward(self, x):
        # segmentation logits
        seg_logits = self.seg(x)  # raw logits
        seg_out = torch.sigmoid(seg_logits)  # convert to [0,1]
        # encoder features for anchors
        feats = self.seg.encoder(x)
        deepest = feats[-1]
        kp_out = self.kp(deepest)
        return seg_out, kp_out


# -----------------------
# Losses & metrics
# -----------------------
def dice_loss(pred, target, eps=1e-6):
    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)
    inter = (pred * target).sum(1)
    denom = pred.sum(1) + target.sum(1)
    loss = 1 - ((2*inter + eps) / (denom + eps))
    return loss.mean()

def iou_numpy(pred_mask, gt_mask, thr=0.5):
    p = (pred_mask >= thr).astype('uint8')
    g = (gt_mask >= 0.5).astype('uint8')
    inter = (p & g).sum()
    union = (p | g).sum()
    return inter/(union+1e-9)

# -----------------------
# Train / validate
# -----------------------
def train_epoch(model, loader, optimizer, device, kp_weight):
    model.train()
    running = 0.0
    for imgs, masks, anchors, fnames in tqdm(loader):
        imgs = imgs.to(device); masks = masks.to(device); anchors = anchors.to(device)
        optimizer.zero_grad()
        seg_pred, kp_pred = model(imgs)
        bce = F.binary_cross_entropy(seg_pred, masks)
        dloss = dice_loss(seg_pred, masks)
        kp_loss = F.mse_loss(kp_pred, anchors)
        loss = bce + dloss + kp_weight * kp_loss
        loss.backward()
        optimizer.step()
        running += loss.item()
    return running / len(loader)

def validate(model, loader, device):
    model.eval()
    ious = []
    kp_errs = []
    with torch.no_grad():
        for imgs, masks, anchors, fnames in tqdm(loader):
            imgs = imgs.to(device)
            seg_pred, kp_pred = model(imgs)
            seg_np = seg_pred.cpu().numpy()[:,0]
            mask_np = masks.numpy()[:,0]
            kp_np = kp_pred.cpu().numpy(); anchors_np = anchors.numpy()
            for p, g, kp_p, kp_g in zip(seg_np, mask_np, kp_np, anchors_np):
                ious.append(iou_numpy((p>0.5).astype('uint8'), g.astype('uint8')))
                if (kp_g[0] >= 0):
                    # compute angle error or L2
                    kp_errs.append(np.linalg.norm(kp_p - kp_g))
    return float(np.mean(ious)), (float(np.mean(kp_errs)) if kp_errs else None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', required=True)
    parser.add_argument('--val_csv', required=True)
    parser.add_argument('--anchors_csv', default='anchors_gt.csv')
    parser.add_argument('--images_dir', default='images/train')
    parser.add_argument('--masks_dir', default='masks/train')
    parser.add_argument('--input_h', type=int, default=320)
    parser.add_argument('--input_w', type=int, default=576)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--kp_weight', type=float, default=1.0)
    parser.add_argument('--encoder', default='resnet34')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device:", device)

    train_trans = get_transforms(train=True, size=(args.input_h, args.input_w))
    val_trans = get_transforms(train=False, size=(args.input_h, args.input_w))

    train_ds = RunwayDataset(args.train_csv, images_dir='images/train', masks_dir='masks/train',
                             anchors_csv=args.anchors_csv, transform=train_trans, input_size=(args.input_h,args.input_w))
    val_ds = RunwayDataset(args.val_csv, images_dir='images/train', masks_dir='masks/train',
                           anchors_csv=args.anchors_csv, transform=val_trans, input_size=(args.input_h,args.input_w))

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=max(1,args.batch//2), shuffle=False, num_workers=2, pin_memory=True)

    model = RunwayModel(encoder_name=args.encoder).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_iou = 0.0

    for epoch in range(1, args.epochs+1):
        print(f"Epoch {epoch}/{args.epochs}")
        tr_loss = train_epoch(model, train_loader, optimizer, device, kp_weight=args.kp_weight)
        val_iou, val_kp_err = validate(model, val_loader, device)
        print(f"TrainLoss {tr_loss:.4f}  ValIoU {val_iou:.4f}  ValKPerr {val_kp_err}")
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), 'best_runway.pth')
            print("Saved best_runway.pth")
    # export ONNX (load best model)
    try:
        model.load_state_dict(torch.load('best_runway.pth', map_location=device))
        model.eval()
        dummy = torch.randn(1,3,args.input_h,args.input_w).to(device)
        torch.onnx.export(model, dummy, "runway_model.onnx", opset_version=11,
                          input_names=['input'], output_names=['seg','kp'],
                          dynamic_axes={'input':{0:'batch'}, 'seg':{0:'batch'}, 'kp':{0:'batch'}})
        print("Exported runway_model.onnx")
    except Exception as e:
        print("ONNX export failed:", e)

if __name__ == '__main__':
    main()
