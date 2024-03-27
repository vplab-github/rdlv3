import argparse
import os
import random
import time

import cv2
import numpy as np
import torch

from transforms.test_transforms import *
from utils.utils import *
from utils.Visualize import *




def cropped_segmentation_test(image, model, device, name):
    h,w,c = image.shape
    h = h // 4
    w = w // 4

    output_image = np.zeros((image.shape))
    output_mask = np.zeros((image.shape[0], image.shape[1], 1)).astype(np.uint8)

    start = time.time()

    for i in range(4):
        for j in range(4):
            sub_img = image[i * h : (i + 1) * h, j * w : (j + 1) * w ,:].copy()
            sub_out, mask = segmentation_test(sub_img, model, device, name)
            output_image[i * h : (i + 1) * h, j * w : (j + 1) * w ,:] = sub_out
            output_mask[i * h : (i + 1) * h, j * w : (j + 1) * w ] = mask * 255
    
    end = time.time()
    time_taken = end - start
    print(f"Total Rutime : {time_taken} sec")

    name = name.split(".")[0]
    # save image 
    # cv2.imwrite(f"{name}-result.tif", output_mask)
    # show final result
    # cv2.destroyAllWindows()
    key = ShowImage(np.hstack((image, output_image)), name=name)
    
    return output_image, key


def segmentation_test(image, model, device, name):
    image_viz = image.copy()

    # pre process the image
    train_image = PreProcessImage(image)

    start = time.time()

    model = model.eval()

    x_tensor = torch.from_numpy(train_image).to(device).unsqueeze(0)
    with torch.autograd.no_grad():
        pred_mask = model(x_tensor)
    pred_mask = pred_mask.detach().squeeze().cpu().numpy()

    end = time.time()
    time_taken = end - start
    print(f"Rutime : {time_taken} sec")

    if pred_mask.ndim == 2:
        pred_mask = np.expand_dims(pred_mask, axis=0)
    # Convert pred_mask from `CHW` format to `HWC` format
    pred_mask = np.transpose(pred_mask, (1, 2, 0))
    pred_mask = np.round(pred_mask).astype(np.uint8)
    pred_mask = PostProcessMask(pred_mask, image)

    liq_image = image_viz.copy()

    for i in range(pred_mask.shape[2]):
        # overlay on mask
        obj_mask = pred_mask[:, :, i]
        # image_viz = OverlayMasks(image_viz, GetGradientMask(obj_mask), color=COLOR_PALLET[i], alpha = 0.9)
        image_viz = OverlayMasks(image_viz, obj_mask, color=COLOR_PALLET[i], alpha=0.9)
        # break

    cv2.destroyWindow(name)
    return image_viz, pred_mask


def RunPath(path, model, device, shufle=True):
    # get all images
    image_paths = GetAllImagesDir(path)
    if shufle:
        random.shuffle(image_paths)
    
    path_len = len(image_paths)
    print(f"Total images found in {path} is {path_len}")
    i = 0
    back = False
    # run model on images
    while i < path_len and i >= 0:
        image_path = image_paths[i]
        file_name = GetFileNamefromPath(image_path)
        image = cv2.imread(image_path)
        if image is None:
            print("No image")
            offset = -1 if back else 1
            i = i + offset
            continue
        print(f"Loaded image {file_name} of size {image.shape}")
        image_viz, key = cropped_segmentation_test(image, model, device, file_name)
        if key == ord("b"):
            back = True
            i = i - 1
            print("back!")
            continue
        back = False
        i = i + 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", "-i", type=str, default="")
    parser.add_argument("--model", "-m", type=str, default="./logs/")

    opt_parser = parser.parse_args()
    image_path = opt_parser.image
    model_path = opt_parser.model

    device = GetDevice()
    print("Using device : {}".format(device))

    # load best saved model checkpoint from the current run
    if os.path.exists(model_path):
        model = torch.load(model_path, map_location="cpu")
        # model = torch.load(model_path)
        model = model.to(device)
        print(f"Loaded Model from {model_path}")
    else:
        print("No Model Found")
        exit(-1)

    model = model.eval()
    RunPath(image_path, model, device)
    cv2.destroyAllWindows()
