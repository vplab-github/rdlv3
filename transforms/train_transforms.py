import albumentations as album

import cfg


class TrainingAugumentation(object):
    def training_augmentation(self):
        train_transform = [
            album.LongestMaxSize(
                max_size=cfg.width, interpolation=1, always_apply=True
            ),
            album.PadIfNeeded(
                min_height=cfg.height,
                min_width=cfg.width,
                always_apply=True,
                border_mode=0,
            ),
            album.Resize(cfg.height, cfg.width),
            album.RandomCrop(height=cfg.height, width=cfg.width, always_apply=True),
            album.Flip(p=0.5),
            album.Transpose(p=0.5),
            album.ShiftScaleRotate(
                shift_limit=0.1, scale_limit=0.1, rotate_limit=5, p=0.5, border_mode=0
            ),
            album.RGBShift(r_shift_limit=50, g_shift_limit=50, b_shift_limit=50, p=0.5),
            album.RandomBrightnessContrast(
                brightness_limit=0.3, contrast_limit=0.3, p=0.5
            ),
            album.OneOf(
                [
                    album.MotionBlur(p=1),
                    album.OpticalDistortion(p=1),
                    album.GaussNoise(p=1),
                ],
                p=0.8,
            ),
        ]
        return album.Compose(train_transform)

    def __call__(self, image, mask):
        sample = self.training_augmentation()(image=image, mask=mask)
        image, mask = sample["image"], sample["mask"]
        return image, mask

    def __repr__(self):
        return self.__class__.__name__


def get_training_augmentation():
    return TrainingAugumentation()
