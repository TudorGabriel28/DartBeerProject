import argparse
from yacs.config import CfgNode as CN
import os.path as osp
import os
from dataloader import get_splits
import cv2
import numpy as np
from time import time
from dataset.annotate import draw, get_dart_scores
import pickle
from predict import bboxes_to_xy
import datetime


def predict_stream(yolo):

    cam = cv2.VideoCapture(0)
    print(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(cam.get(cv2.CAP_PROP_FPS))
    i = 0

    while True:
        check, frame = cam.read()
        # Resize frame to 800x800
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # img = img[50:1000, 400:1000]
        img = cv2.resize(img, (800, 800))
        bboxes = yolo.predict(img)
        preds = bboxes_to_xy(bboxes, 3)
        xy = preds
        xy = xy[xy[:, -1] == 1]
        img = draw(
            cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
            xy[:, :2],
            cfg,
            circles=False,
            score=True,
        )
        cv2.imshow("video", img)

        key = cv2.waitKey(1)
        if key == "z":
            break

    cam.release()
    cv2.destroyAllWindows()


def collect_data():
    # Open a connection to the webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        # Get the original dimensions
        h, w = frame.shape[:2]

        # Calculate the center crop dimensions
        if w > h:
            new_w = h
            new_h = h
            x_offset = (w - h) // 2
            y_offset = 0
        else:
            new_w = w
            new_h = w
            x_offset = 0
            y_offset = (h - w) // 2

        # Crop the image to the calculated dimensions
        frame_cropped = frame[y_offset : y_offset + new_h, x_offset : x_offset + new_w]

        # Resize the cropped image to 800x800
        frame_resized = cv2.resize(frame_cropped, (800, 800))

        # Display the resulting frame
        cv2.imshow("Press Space to Take Photo", frame_resized)

        # Wait for the user to press a key
        key = cv2.waitKey(1) & 0xFF

        # If the user presses the space bar, take a photo
        if key == ord(" "):
            # Get the current date
            now = datetime.datetime.now()
            date_str = now.strftime("utrecht_%m_%d_%Y")

            # Create the directory if it does not exist
            save_dir = os.path.join("dataset/images", date_str)
            os.makedirs(save_dir, exist_ok=True)

            # Save the photo
            photo_path = os.path.join(save_dir, f"photo_{now.strftime('%H_%M_%S')}.jpg")
            cv2.imwrite(photo_path, frame_resized)
            print(f"Photo saved at {photo_path}")

        # If the user presses the 'q' key, exit the loop
        elif key == ord("q"):
            break

    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    from train import build_model

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cfg", default="deepdarts_utrecht")
    parser.add_argument("-cd", "--collect-data", default=False)
    args = parser.parse_args()

    if args.collect_data == "True":
        collect_data()
    else:
        cfg = CN(new_allowed=True)
        cfg.merge_from_file(osp.join("configs", args.cfg + ".yaml"))
        cfg.model.name = args.cfg

        yolo = build_model(cfg)
        yolo.load_weights(
            osp.join("models", args.cfg, "weights"), cfg.model.weights_type
        )

        predict_stream(yolo)
