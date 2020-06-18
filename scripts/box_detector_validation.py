# validate YOLOv3 box detector
# calculate mean intersection over union for all test images

import os
import sys
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import sys
from prediction_utils import *
from shapely.geometry import Polygon


src_path = os.path.join('..', 'YOLOv3', 'TrainYourOwnYOLO', "2_Training", "src")
utils_path = os.path.join('..', 'YOLOv3', 'TrainYourOwnYOLO', "Utils")

sys.path.append(src_path)
sys.path.append(utils_path)

# more imports here
from keras_yolo3.yolo import YOLO
from utils import detect_object

# Set up folder names for default values
data_folder = os.path.join('..','YOLOv3','TrainYourOwnYOLO', "Data")
image_folder = os.path.join(data_folder, "Source_Images")
image_test_folder = os.path.join(image_folder, "Test_Images")
model_folder = os.path.join(data_folder, "Model_Weights")
model_weights = os.path.join(model_folder, "trained_weights_final_ck2000.h5")
model_classes = os.path.join(model_folder, "data_classes.txt")
anchors_path = os.path.join(src_path, "keras_yolo3", "model_data", "yolo_anchors.txt")

# define YOLO detector
yolo = YOLO(
    **{
        "model_path": model_weights,
        "anchors_path": anchors_path,
        "classes_path": model_classes,
        "score": 0.25,
        "gpu_num": 1,
        "model_image_size": (416, 416),
    }
)

# set test image directory
test_dir = os.path.join('..', 'data', 'test_data')
png_dir = os.path.join(test_dir, 'PNG')
img_files = os.listdir(png_dir)

# load target data
all_targets = np.load(os.path.join(test_dir, 'targets.npy'), allow_pickle=True)[()]

# calculate IoU
def get_IoU(bb1, bb2):
    """ Compute intersection over union """
    a = Polygon([(bb1[0],bb1[1]), (bb1[0], bb1[3]), (bb1[2], bb1[3]), (bb1[2], bb1[1])])
    b = Polygon([(bb2[0],bb2[1]), (bb2[0], bb2[3]), (bb2[2], bb2[3]), (bb2[2], bb2[1])])
    return a.intersection(b).area / a.union(b).area

# set minimum overlap threshold
overlap_threshold = 0.2

sumIoU = 0.0  # this becomes the numerator in the average
counter = 0   # this becomes the denomenator in the average

# Loop through test images
for file in img_files:
    print(f"\nLoading image: {file}")
    img = Image.open(os.path.join(png_dir, file)).convert("RGB")

    # set target for this image
    targets = all_targets[file]

    # Run YOLO detector
    new_img = copy.deepcopy(img)
    YOLO_predictions, new_image = yolo.detect_image(new_img);

    matched_gt = []
    matched_p = []
    # loop through groundtruth boxes
    for targetID, target in enumerate(targets):
        gt_bb = target['bbox']

        # loop through predicted boxes
        maxIoU = 0
        maxID = -1
        for ix, p_bb in enumerate(YOLO_predictions):
            if ix in matched_p:
                continue

            # calculate IoU
            IoU = get_IoU(gt_bb, p_bb)

            # find best score
            if IoU > maxIoU:
                maxIoU = IoU
                maxID = ix

        print(f"Target {targetID}; Max IoU: {maxIoU}")

        # Delete ground truth and predicted boxes from their respective arrays if overlap exceeds threshold
        if maxIoU > overlap_threshold:
            sumIoU += maxIoU
            counter += 1

            matched_gt.append(targetID)
            matched_p.append(maxID)

    # Deal with possibility that a ground truth box didn't have a corresponding predicted box or vice versa
    counter += len(YOLO_predictions) - len(matched_p)
    counter += len(targets) - len(matched_gt)

# compute average
avgIoU = sumIoU / counter


print(f"Average IoU: {avgIoU}")
