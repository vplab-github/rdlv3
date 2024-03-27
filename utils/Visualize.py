import cv2
import numpy as np

COLOR_PALLET = [
    (255, 0, 255),
    (0, 255, 0),
    (0, 0, 255),
    (255, 0, 0),
    (0, 255, 255),
    (255, 255, 0),
]


def OverlayMasks(image, masks, color=(0, 0, 255), alpha=0.3):
    mask = np.round(masks).astype(np.uint8)
    mask = mask[..., np.newaxis]
    out_image = image.copy()
    ColImg = np.zeros(out_image.shape, out_image.dtype)
    ColImg[:, :] = color
    NewMask = cv2.bitwise_and(ColImg, ColImg, mask=mask)
    cv2.addWeighted(NewMask, alpha, out_image, 1.0, 0, out_image)
    # out_image = np.concatenate((image,out_image), axis = 1)
    return out_image


def OverlayMasks(image, masks, color=(0, 0, 255), alpha=0.3):
    mask = np.round(masks).astype(np.uint8)
    mask = mask[..., np.newaxis]
    out_image = image.copy()
    # out_image[masks == 1] = color
    # return out_image
    ColImg = np.zeros(out_image.shape, out_image.dtype)
    ColImg[:, :] = color
    NewMask = cv2.bitwise_and(ColImg, ColImg, mask=mask)
    cv2.addWeighted(NewMask, alpha, out_image, 1.0, 0, out_image)
    # out_image = np.concatenate((image,out_image), axis = 1)
    return out_image


def GetGradientMask(mask):
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=10)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=10)
    # invert the image
    invert = cv2.bitwise_not(mask)
    # use morph gradient
    morph_gradient = cv2.morphologyEx(invert, cv2.MORPH_GRADIENT, kernel, iterations=10)
    return morph_gradient


# helper function for image visualization
def ShowImage(image, name="show", time_wait=0, colab=False):
    if image.ndim == 2:
        image = image[..., np.newaxis]
    if image.shape[0] < image.shape[-1]:
        image = image.transpose((1, 2, 0))
    image = image.astype(np.uint8)
    if np.max(image) == 1:
        image = image * 255
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, image)
    key = cv2.waitKey(time_wait)
    return key
    # cv2.destroyAllWindows()
