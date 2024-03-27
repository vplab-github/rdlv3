import os
import time
from glob import glob

import torch

import cfg


def GetDateTime():
    return time.strftime("%Y-%m-%d-%H-%M-%S")


def GetModelSaveName():
    return f"{GetDateTime()}-{cfg.encoder}-H{cfg.height}-W{cfg.width}.pth"


def GetDevice():
    if cfg.device == "":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    else:
        return torch.device(cfg.device)


def TotalModelParams(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def GetAllImagesDir(in_path="../cropped_dataset_incomplete/"):
    if os.path.isfile(in_path):
        return [in_path]

    out = []
    if os.path.isdir(in_path):
        result = [y for x in os.walk(in_path) for y in glob(os.path.join(x[0], "*"))]
        for path in result:
            ext = os.path.splitext(path)[-1]
            if os.path.isfile(path) and (
                ext == ".bmp"
                or ext == ".jpeg"
                or ext == ".png"
                or ext == ".jpg"
                or ext == ".ppm"
            ):
                out.append(path)
    else:
        return []
    return out


def GetFileNamefromPath(path):
    return os.path.basename(path)
