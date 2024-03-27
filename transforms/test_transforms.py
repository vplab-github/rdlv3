import cv2
import numpy as np

import cfg


class ValidationAugumentation(object):
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def transform(self, image, interpolation=cv2.INTER_LINEAR):
        # resize to desired ratio
        h, w, _ = image.shape
        h_ratio = h / cfg.height
        w_ratio = w / cfg.width
        if h_ratio > w_ratio:
            new_width = int((w / h) * cfg.height)
            new_height = cfg.height
        else:
            new_height = int((h / w) * cfg.width)
            new_width = cfg.width
        image = cv2.resize(image, (new_width, new_height), interpolation=interpolation)

        # pad to cfg.width, cfg.height
        if len(image.shape) == 2:
            image = image[..., np.newaxis]
        h, w, _ = image.shape
        h_pad = int(abs(cfg.height - h) / 2)
        w_pad = int(abs(cfg.width - w) / 2)
        image = cv2.copyMakeBorder(
            image, h_pad, h_pad, w_pad, w_pad, cv2.BORDER_CONSTANT, value=0
        )

        # resize image to cfg.width, cfg.height (for removing int conversion errors)
        image = cv2.resize(image, (cfg.width, cfg.height), interpolation=interpolation)
        if len(image.shape) == 2:
            image = image[..., np.newaxis]
        return image

    def __call__(self, image, mask=None):
        if image is not None:
            image = self.transform(image, interpolation=cv2.INTER_LINEAR)
        if mask is not None:
            mask = self.transform(mask, interpolation=cv2.INTER_NEAREST)
        return image, mask

    def __repr__(self):
        return self.__class__.__name__ + "(size={0})".format((self.height, self.width))


class PreProcessing(object):
    def PreProcessFn(self, x, **kwargs):
        x = x.astype(np.float32)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        input_space = "RGB"
        input_range = [0, 1]

        if input_space == "BGR":
            x = x[..., ::-1].copy()
        if input_range is not None:
            if x.max() > 1 and input_range[1] == 1:
                x = x / 255.0
        if mean is not None:
            mean = np.array(mean)
            x = x - mean
        if std is not None:
            std = np.array(std)
            x = x / std
        return x

    def to_tensor(self, x):
        return x.transpose(2, 0, 1).astype("float32")

    def __call__(self, image, mask=None):
        if image is not None:
            image = self.PreProcessFn(image)
            image = self.to_tensor(image)
        if mask is not None:
            mask = self.to_tensor(mask)
        return image, mask

    def __repr__(self):
        return self.__class__.__name__


def PostProcessMask(mask, image):
    dim_img = image.shape
    side = max(dim_img[0], dim_img[1])

    new_mask = cv2.resize(mask, (side, side), interpolation=cv2.INTER_NEAREST)

    dim_mask = new_mask.shape

    if len(new_mask.shape) == 2:
        new_mask = new_mask[..., np.newaxis]

    diff_x = dim_mask[0] - dim_img[0]
    diff_y = dim_mask[1] - dim_img[1]

    if diff_x > 0:
        diff_x_2 = int(diff_x / 2)
        new_mask = new_mask[diff_x_2:-diff_x_2, :, :]

    if diff_y > 0:
        diff_y_2 = int(diff_y / 2)
        new_mask = new_mask[:, diff_y_2:-diff_y_2, :]

    new_mask = cv2.resize(
        new_mask, (dim_img[1], dim_img[0]), interpolation=cv2.INTER_NEAREST
    )

    if len(new_mask.shape) == 2:
        new_mask = new_mask[..., np.newaxis]

    return new_mask


def get_validation_augmentation():
    return ValidationAugumentation(cfg.height, cfg.width)


def get_preprocessing():
    return PreProcessing()


def PreProcessImage(image):
    out, _ = get_validation_augmentation()(image=image)
    out, _ = get_preprocessing()(image=out)
    return out
