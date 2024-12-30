import os
from ultralytics import YOLO

# Feel free to remove this line (not important for the model)
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


if __name__ == "__main__":
    # Create a new YOLO model from scratch, No pretrained weights are available for now
    # but you can quickly train the model on your custom dataset
    # this new model can work on much smaller
    # dataset then the original models

    model = YOLO("yolo_dinov2_configs/yolo_dinov2_small.yaml")

    # Train test and predict as you will do for any other yolo model
    # in the repository, Please refer to the ultralytics
    # documentation for more details
