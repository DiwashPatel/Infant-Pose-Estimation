#!/usr/bin/env python3

import json
import argparse
from pathlib import Path
import shutil

import cv2
import numpy as np


# ------------------------------------------------------------
# Repository root
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

IMAGE_DIR = ROOT / "data" / "custom" / "predict"
BBOX_DIR = ROOT / "intermediate" / "bbox"
OUTPUT_DIR = ROOT / "visualizations"

KEYPOINT_DIR = ROOT / "intermediate" / "keypoints"
KEYPOINT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Automatically locate prediction json
# ------------------------------------------------------------

matches = list(ROOT.glob("output/**/results/keypoints*_results*.json"))

if len(matches) == 0:
    raise FileNotFoundError(
        "Could not find keypoints results JSON under output/**/results/"
    )

KEYPOINT_JSON = matches[0]

print(f"Using keypoint file:\n{KEYPOINT_JSON}\n")


# ------------------------------------------------------------
# Arguments
# ------------------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument(
    "--mode",
    choices=["bbox", "keypoints", "both"],
    default="both",
)

parser.add_argument(
    "--kp_threshold",
    type=float,
    default=0.2,
)

args = parser.parse_args()


OUTPUT_DIR.mkdir(exist_ok=True)


# ------------------------------------------------------------
# Load predictions
# ------------------------------------------------------------

with open(KEYPOINT_JSON, "r") as f:
    predictions = json.load(f)

prediction_dict = {}

for item in predictions:
    prediction_dict[str(item["image_id"])] = item

print(f"Loaded {len(prediction_dict)} predictions.")


# ------------------------------------------------------------
# Save one keypoint json per image
# ------------------------------------------------------------

repo_root = Path(__file__).resolve().parent.parent

KEYPOINT_DIR = repo_root / "intermediate" / "keypoints"
KEYPOINT_DIR.mkdir(parents=True, exist_ok=True)

for image_id, pred in prediction_dict.items():

    
    annotations = pred["keypoints"][:]   # Copy the flat list

    # Set every confidence value to 2.0
    for i in range(2, len(annotations), 3):
        annotations[i] = 2.0

    annotation_info = {
        "annotations": annotations
    }
    
    out_file = KEYPOINT_DIR / f"{image_id}_keypoints.json"

    with open(out_file, "w") as f:
        json.dump(annotation_info, f)


# ------------------------------------------------------------
# Optionally copy to syn_generation
# ------------------------------------------------------------

syn_keypoint_dir = (
    repo_root /
    "syn_generation" /
    "data" /
    "keypoints"
)

if syn_keypoint_dir.exists():

    for json_file in KEYPOINT_DIR.glob("*_keypoints.json"):
        shutil.copy2(json_file, syn_keypoint_dir)

    print(f"Copied {len(list(KEYPOINT_DIR.glob('*_keypoints.json')))} keypoint files to")
    print(f"  {syn_keypoint_dir}")




import shutil

# Define the destination directory for synthetic images
syn_image_dir = (
    repo_root /
    "syn_generation" /
    "data" /
    "images"
)

if syn_image_dir.exists():
    # Find all .png and .jpg files in your source image directory
    image_files = list(IMAGE_DIR.glob("*.png")) + list(IMAGE_DIR.glob("*.jpg"))
    
    # Copy each image to the destination
    for img_file in image_files:
        shutil.copy2(img_file, syn_image_dir)

    print(f"Copied {len(image_files)} image files to")
    print(f"  {syn_image_dir}")
# ------------------------------------------------------------
# Process every bbox json
# ------------------------------------------------------------

for bbox_json in sorted(BBOX_DIR.glob("*.json")):

    with open(bbox_json) as f:
        bbox_info = json.load(f)

    image_name = bbox_info["image"]
    image_id = Path(image_name).stem

    image_path = IMAGE_DIR / image_name

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Cannot open {image_path}")
        continue

    # --------------------------------------------------------
    # Bounding box
    # --------------------------------------------------------

    if args.mode in ("bbox", "both"):

        x1, y1, x2, y2 = bbox_info["bbox"]

        cv2.rectangle(
            image,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 0, 255),
            2,
        )

    # --------------------------------------------------------
    # Keypoints
    # --------------------------------------------------------

    if args.mode in ("keypoints", "both"):

        pred = prediction_dict.get(image_id)

        if pred is not None:

            kpts = np.array(pred["keypoints"]).reshape(-1, 3)

            for x, y, score in kpts:

                if score < args.kp_threshold:
                    continue

                cv2.circle(
                    image,
                    (int(round(x)), int(round(y))),
                    4,
                    (0, 255, 0),
                    -1,
                )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    if args.mode == "bbox":
        suffix = "_bbox"

    elif args.mode == "keypoints":
        suffix = "_keypoints"

    else:
        suffix = "_bbox_keypoints"

    out_file = OUTPUT_DIR / f"{image_id}{suffix}.jpg"

    cv2.imwrite(str(out_file), image)

    print(f"Saved {out_file}")

print("\nFinished.")