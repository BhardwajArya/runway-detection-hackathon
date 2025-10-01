Runway Detection Hackathon

End-to-end deep learning pipeline for runway detection and localization developed during the Honeywell Aerospace Hackathon at Manipal.

The project performs:

Segmentation of runway areas (IoU evaluation).

Anchor regression to predict runway edge lines.

Boolean scoring to check correctness of predicted runway center.

Inference service that accepts an uploaded image and returns predictions.

Project Structure
runway_project/
│
├── images/                # Original dataset images
│   ├── train/
│   ├── val/
│   └── test/
│
├── masks/                 # Generated binary masks
│   ├── train/
│   ├── val/
│   └── test/
│
├── predictions/           # Model outputs
│   ├── masks/             # Predicted segmentation masks
│   ├── anchors_pred.csv   # Predicted anchors
│   └── submission.csv     # Final submission file
│
├── anchors_gt.csv         # Ground-truth anchors from preprocessing
├── train_files.csv        # Training split
├── val_files.csv          # Validation split
│
├── deep_runway.py         # Training script
├── inference.py           # Offline inference script
├── compute_scores.py      # Evaluation script (IoU, anchor score, boolean score)
├── inference_server.py    # Flask app for image upload + prediction
└── README.md

Dataset

We used the FS2020 Runway Dataset from Kaggle:
https://www.kaggle.com/datasets/relufrank/fs2020-runway-dataset

Setup

Clone the repo and install dependencies (conda recommended):

git clone https://github.com/<your-username>/runway-detection-hackathon.git
cd runway-detection-hackathon

conda create -n runway python=3.10 -y
conda activate runway

pip install -r requirements.txt


If using GPU: install PyTorch with CUDA from official instructions
.

Preprocessing

Convert JSON annotations into masks and anchors:

python json_to_masks.py --json <labels.json> --images_dir images/train --out_dir masks/train --width 640 --height 360
python create_anchors_from_masks.py --masks_dir masks/train --out anchors_gt.csv
python make_splits.py --images_dir images/train --out_train train_files.csv --out_val val_files.csv

Training

Example training run (on CPU with reduced resolution for speed):

python deep_runway.py \
  --train_csv train_files.csv \
  --val_csv val_files.csv \
  --anchors_csv anchors_gt.csv \
  --epochs 8 \
  --batch 4 \
  --input_h 180 --input_w 320 \
  --kp_weight 8.0


Outputs:

best_runway.pth (best PyTorch weights)

Validation IoU and anchor error scores printed per epoch

Inference

Run predictions on the test set:

python inference.py


Outputs:

Predicted masks in predictions/masks/

Predicted anchors in predictions/anchors_pred.csv

Final submission.csv in correct hackathon format

Evaluation

Compute IoU, anchor score, and boolean score on validation data:

python compute_scores.py


Outputs:

metrics_per_image.csv (per-image results)

Mean IoU, anchor score, boolean score

Inference Service

Start a local server to upload an image and get predictions:

python inference_server.py


Upload an image only → returns predicted anchors and saves a mask.

Upload an image + ground truth mask → returns IoU, anchor score, and boolean score.

Example request:

curl -X POST -F "image=@test_img.png" -F "gt_mask=@gt_mask.png" http://localhost:5000/predict

Results

Validation IoU after 2 epochs on CPU: ~0.35

Anchor and boolean scores improve significantly with more epochs and higher resolution training

Acknowledgements

Dataset: FS2020 Runway Dataset on Kaggle

Segmentation backbone: segmentation-models-pytorch

Built for the Honeywell Aerospace Hackathon at Manipal
