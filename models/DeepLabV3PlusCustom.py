import segmentation_models_pytorch as smp
import torch

import cfg


class DeepLabV3PlusCustom(torch.nn.Module):
    def __init__(
        self,
        encoder=cfg.encoder,
        encoder_weights=cfg.encoder_weights,
        num_classes=1,
        decoder_channels=cfg.decoder_channels,
    ):
        super(DeepLabV3PlusCustom, self).__init__()

        self.num_classes = num_classes
        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            classes=num_classes,
            activation="sigmoid",
            decoder_channels=decoder_channels,
        )

    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == "__main__":
    import time

    from utils.utils import *

    model = DeepLabV3PlusCustom(num_classes=2)
    model = model.eval()
    print(model)
    data = torch.rand(
        (1, 3, cfg.height, cfg.width), dtype=torch.float, requires_grad=False
    )
    print(f"total model params for training : {TotalModelParams(model):,}")
    time_list = []

    for i in range(10):
        start = time.time()
        out = model(data)
        end = time.time()
        time_taken = end - start
        time_list.append(time_taken)
        print(time_taken)

    mean_time = sum(time_list) / len(time_list)
    print(f"Average time taken : {mean_time}")
    for ou in out:
        print(ou.size())
    print(len(out))

    # torch.save(model, './logs/best_model.pth')
    # print('Model saved!')
