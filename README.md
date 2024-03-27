# Rdlv3

This repo contains code for training and testing of Cell Segmenation Algorithm. We use the DeepLabV3+ with Resnet18 as described in the [paper](https://arxiv.org/abs/1802.02611) as our model, on the cropped cell dataset.


### Steps to Run
1. Clone the Repo
2. Install all the required python3 packages using
```
pip3 install -r requirements.txt
```
3. To test our model on the test images run the `test_cropped.py` as
``` 
python3 test_cropped.py -i <path to image / images dir> -m <path to model> 
```
an example being:
``` 
python3 test_cropped.py -i ./test_images/ -m logs/best_model.pth
```
to run the cell detection demo. 

4. To get the accuracy and the detection scores run the `evaluate.py` as
``` 
python3 evaluate.py -g <path to gt image> -r <path to result image>
```
an example being:
``` 
python evaluate.py -g test_images/15188.tif -r test_images/15188-result.tif
```
to run the evaluation. 

### Training custom model
1. Download the `cropped cell dataset` from [here](https://drive.google.com/file/d/1xeYq03wAL4_kO9OUoyt79i1nYhtOsUCr/view?usp=sharing). You can use [anylabeling](https://github.com/vietanhdev/anylabeling) to open, view and edit the dataset.
2. Edit the `cfg.py` to suit your needs. Also do not forget to update the `dataset_dir`.
3. To train the model run.
``` 
python3 train.py 
```
4. The trained model is saved in `logs/`.
