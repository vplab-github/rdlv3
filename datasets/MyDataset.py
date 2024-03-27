import glob
import json
import os
import random

import cv2
import numpy as np
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(
        self,
        dataset_dir="../cropped_cell_dataset/",
        classes=["cell"],
        train=True,
        augumentation=None,
        pre_processing=None,
    ):
        self.dataset_dir = dataset_dir
        self.augumentation = augumentation
        self.pre_processing = pre_processing
        self.train = train
        self.classes = classes

        self.all_data = glob.glob(
            os.path.join(self.dataset_dir, "labels/*.json"), recursive=True
        )
        random.Random(1234).shuffle(self.all_data)

        total_data = len(self.all_data)
        train_data = int(total_data * 0.75)

        self.data_labels_path = []
        if self.train:
            self.data_labels_path = self.all_data[0:train_data]
        else:
            self.data_labels_path = self.all_data[train_data:]

        self.data_images_path = []
        for paths in self.data_labels_path:
            f = open(paths)
            data = json.load(f)
            image_path = data["imagePath"]
            image_path = image_path.replace("../", "").replace("..\\", "").replace("\\", "/")
            image_path = os.path.join(self.dataset_dir, image_path)
            self.data_images_path.append(image_path)

    def __len__(self):
        return len(self.data_labels_path)

    def __getitem__(self, index):
        # get path for current index
        label_path = self.data_labels_path[index]
        img_path = self.data_images_path[index]
        # print(img_path)

        # read image
        image = cv2.imread(img_path)

        # one-hot-encode the mask
        mask, ignore = self.MasksFromJson(image, label_path, self.classes)
        image[ignore[:, :, 0] == 1] = (255, 255, 255)

        image, mask = self.ProcessImageMasks(image, mask)

        return image, mask

    @classmethod
    def MasksFromJson(self, image, json_path, labels=[]):
        f = open(json_path)
        h, w, _ = image.shape
        num_classes = len(labels)
        data = json.load(f)
        masks = np.zeros((h, w, num_classes), dtype=np.uint8)
        ignore = np.zeros((h, w, 1), dtype=np.uint8)

        for j, label in enumerate(labels):
            cur_mask = masks[:, :, j].copy()
            for i, shape in enumerate(data["shapes"]):
                if "label" not in shape.keys():
                    continue

                cur_label = shape["label"]
                if cur_label == label:
                    mask = cur_mask.copy()
                    points = np.array(shape["points"], dtype=np.int32)
                    cv2.fillPoly(mask, [points], 1)
                    cur_mask = cv2.bitwise_or(cur_mask, mask)

                # redundant code, todo: need to make it efficient !
                if cur_label == "ignore":
                    points = np.array(shape["points"], dtype=np.int32)
                    cv2.fillPoly(ignore, [points], 1)

            masks[:, :, j] = cur_mask
        return masks, ignore

    def ProcessImageMasks(self, image, mask):
        # apply augumentations
        if self.augumentation is not None:
            image, mask = self.augumentation(image=image, mask=mask)

        # apply preprocessing
        if self.pre_processing:
            image, mask = self.pre_processing(image=image, mask=mask)
        return image, mask

    def getAllClassesList(self):
        classes = []
        for paths in self.data_labels_path:
            f = open(paths)
            data = json.load(f)

            for i, shape in enumerate(data["shapes"]):
                if "label" not in shape.keys():
                    continue

                cur_label = shape["label"]
                if cur_label == "ignore":
                    continue
                if cur_label not in classes:
                    classes.append(cur_label)
        return classes


if __name__ == "__main__":
    from transforms.test_transforms import (
        get_preprocessing,
        get_validation_augmentation,
    )
    from transforms.train_transforms import get_training_augmentation
    from utils.Visualize import OverlayMasks, ShowImage

    dataset = MyDataset(
        train=True, augumentation=get_training_augmentation(), pre_processing=None
    )
    print(f"Total Dataset size : {len(dataset)}")

    for image, mask in dataset:
        print(image.shape)
        new_image = OverlayMasks(image, mask[:, :, 0], alpha=0.9, color=(255, 0, 255))
        key = ShowImage(new_image)
        if key == ord('s'):
            cv2.imwrite("output.png",new_image)
