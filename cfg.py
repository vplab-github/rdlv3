# dataset params
# Put training data folder here.
dataset_dir = "../cropped_cell_dataset/"
# classes are based on annotation names
classes = ["cell"]

sq = 640
height = int(sq)
width = int(sq)

# Dataloader params
train_batch_size = 2
test_batch_size = 2

# Train params
num_epochs = 100
learning_rate = 0.0006
# "" - Use best device available
# "cpu" - cpu
# "cuda" - gpu
# "mps" - apple M GPU
device = ""
save_dir = "./logs/"


# Model params
# For a list of available encoders, and pre-trained weights
# click the link below
# https://smp.readthedocs.io/en/latest/encoders.html
# encoder = "timm-mobilenetv3_small_minimal_100"
encoder = "resnet18"
encoder_weights = "imagenet"
decoder_channels = 256
