#!/usr/bin/env python3

import json
import argparse
from pathlib import Path

import cv2
import numpy as np


# ------------------------------------------------------------
# Repository root
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

IMAGE_DIR = ROOT / "data" / "custom" / "predict"
BBOX_DIR = ROOT / "intermediate" / "bbox"
OUTPUT_DIR = ROOT / "visualizations"


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