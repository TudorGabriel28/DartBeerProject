import argparse
from yacs.config import CfgNode as CN
import os.path as osp
import os
import cv2
import numpy as np
from time import sleep
from dataset.annotate import draw, get_dart_scores
import json
from predict import bboxes_to_xy


def predict_stream(yolo, cfg, img_paths, output_json="results.json"):
    results = []

    for img_path in img_paths:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (800, 800))

        bboxes = yolo.predict(img)
        preds = bboxes_to_xy(bboxes, 3)
        
        if preds.shape[0] >= 7:  # Ensure there are at least 7 points (4 calibration + 3 darts)
            print(f"Predictions for image {img_path}: {preds}")
            scores = get_dart_scores(preds, cfg, numeric=True)
            print(f"Scores for image {img_path}: {scores}")
            
            for i in range(4, 7):  # Points 5, 6, and 7 (index 4, 5, 6)
                xy = preds[i]
                if xy[2] == 1:  # Check if the dart point is valid
                    result = {"x": float(xy[0]), "y": float(xy[1]), "score": int(scores[i - 4])}
                    print(f"Predicted score for dart {i-3} in image {img_path}: {result['score']}")
                    results.append(result)
                else:
                    print(f"Invalid dart point in image {img_path}: {xy}")
        else:
            print(f"Not enough points for image {img_path}: {preds.shape[0]}")

        sleep(3)  # Wait for 3 seconds before processing the next image

    with open(output_json, "w") as json_file:
        json.dump(results, json_file, indent=4)
    print(f"Saved results to {output_json}")


if __name__ == "__main__":
    from train import build_model

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cfg", default="deepdarts_utrecht")
    args = parser.parse_args()

    cfg = CN(new_allowed=True)
    cfg.merge_from_file(osp.join("configs", args.cfg + ".yaml"))
    cfg.model.name = args.cfg

    yolo = build_model(cfg)
    yolo.load_weights(
        osp.join("models", args.cfg, "weights"), cfg.model.weights_type
    )

    # Define the paths to the 3 images
    img_paths = [
        "dataset/cropped_images/800/utrecht_12_22_2024_val/photo_12_06_40.jpg",
        "dataset/cropped_images/800/utrecht_12_22_2024_val/photo_12_06_48.jpg",
        "dataset/cropped_images/800/utrecht_12_22_2024_val/photo_12_06_57.jpg",
    ]

    predict_stream(yolo, cfg, img_paths)