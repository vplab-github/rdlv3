import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader

import cfg
from datasets.MyDataset import MyDataset as my_dataset
from models.DeepLabV3PlusCustom import DeepLabV3PlusCustom
from transforms.test_transforms import *
from transforms.train_transforms import *
from utils.utils import *

# get DeepLabV3
model_path = os.path.join(cfg.save_dir, GetModelSaveName())
model = DeepLabV3PlusCustom(num_classes=len(cfg.classes))
device = GetDevice()

# Get train and val dataset instances
train_dataset = my_dataset(
    dataset_dir=cfg.dataset_dir,
    classes=cfg.classes,
    train=True,
    augumentation=get_training_augmentation(),
    pre_processing=get_preprocessing(),
)

test_dataset = my_dataset(
    dataset_dir=cfg.dataset_dir,
    classes=cfg.classes,
    train=False,
    augumentation=get_validation_augmentation(),
    pre_processing=get_preprocessing(),
)

print(len(train_dataset))

# Get train and val data loaders
train_loader = DataLoader(
    train_dataset, batch_size=cfg.train_batch_size, shuffle=True, drop_last=True
)
test_loader = DataLoader(
    test_dataset, batch_size=cfg.test_batch_size, shuffle=False, drop_last=True
)

# define loss function
loss = smp.utils.losses.DiceLoss() + smp.utils.losses.BCELoss()
# loss = smp.utils.losses.DiceLoss()
# define metrics
metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
]
# define optimizer
optimizer = torch.optim.Adam(
    [
        dict(params=model.parameters(), lr=cfg.learning_rate),
    ]
)

print("Loaded Datasets ...")
print("Height and Width : {}".format((cfg.height, cfg.width)))
print("Total Train Data Size : {}".format(len(train_dataset)))
print("Total Test Data Size : {}".format(len(test_dataset)))
print(f"Train Batch train size : {cfg.train_batch_size}")
print(f"Test Batch train size : {cfg.test_batch_size}")
print("Training for : {}".format(cfg.classes))
print(f"Total Epochs : {cfg.num_epochs}")
print(f"Using device : {device}")
print(f"Learning rate : {cfg.learning_rate}")
print(f"Using Encoder : {cfg.encoder}")
print(f"Total model params for training : {TotalModelParams(model):,}")
print(f"Model save path : {model_path}")


train_epoch = smp.utils.train.TrainEpoch(
    model,
    loss=loss,
    metrics=metrics,
    optimizer=optimizer,
    device=device,
    verbose=True,
)

test_epoch = smp.utils.train.ValidEpoch(
    model,
    loss=loss,
    metrics=metrics,
    device=device,
    verbose=True,
)

best_iou_score = 0.30
train_logs_list, test_logs_list = [], []

all_test = my_dataset(
    dataset_dir=cfg.dataset_dir,
    classes=cfg.classes,
    train=False,
    augumentation=None,
    pre_processing=None,
)

for i in range(0, cfg.num_epochs):
    # Perform training & testing
    print("\nEpoch: {}".format(i))
    train_logs = train_epoch.run(train_loader)
    test_logs = test_epoch.run(test_loader)

    # for image , _ in all_test:
    #     image_viz, key = segmentation_test(image, model, device, "test")
    #     if key == ord("d"):
    #         break
        # cv2.imshow("test", image_viz)
        # cv2.waitKey(0)
    train_logs_list.append(train_logs)
    test_logs_list.append(test_logs)

    # Save model if a better val IoU score is obtained
    if "iou_score" in test_logs:
        if best_iou_score < test_logs["iou_score"]:
            best_iou_score = test_logs["iou_score"]
            torch.save(model, model_path)
            print("Model saved!")

torch.save(model, model_path)
print("Model saved!")
