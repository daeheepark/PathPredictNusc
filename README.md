# Multimodel Trajectory Prediction
[paper : https://arxiv.org/abs/1809.10732]

## Install Requirement using pip
```
pip install -r requirement.txt
```

## Generate Dataset (Raster image, state vector, gt trajectory) from Nuscenes dataset

You should download [**Nuscenes/Fulldataset(v1.0)/Trainval/Metadata**](https://www.nuscenes.org/download) (Direct download link is [here](https://s3.ap-southeast-1.amazonaws.com/asia.data.nuscenes.org/public/v1.0/v1.0-trainval_meta.tgz?AWSAccessKeyId=AKIA6RIK4RRMFUKM7AM2&Signature=I11deucpmFyJZ0YiQhfPnPtUfQQ%3D&Expires=1601281058))

Unzip downloaded meta filt to [DATAROOT] then run:

```
python gen_dataset.py --dataroot [DATAROOT] --split [SPLIT]
```
[SPLIT] should be one of 'train' or 'train_val' or 'val'

* 'train' : training dataset : 32k 
* 'train_val' : validation dataset : 8k
* 'val' : test dataset : 


## Train MTP model
```
python train_mtp.py --name [NAME] 
```
checkpoint of experiment is saved in './exps/[NAME]'

You can add more argument such as optimizer, backbone, and so on (Refer to train_mtp.py)

## Visualize Trained model from checkpoint
```
visdom
```
run in another window:
```
python visualize.py --name [NAME] --ep [EP]
```
It loads saved weight from checkpoint of experiment named [NAME]. 

[EP] should be 'best' or 'last' or a specific epoch (integer)

* 'best' : w/ lowest validation loss
* 'last' : w/ last epoch
* specific epoch : 1, 10, ...

Generated trajectories and its logits are displayed on [visdom](https://github.com/facebookresearch/visdom) server.

You can see the results at '[YOUR_IP_ADDRESS]:8097' (default port of visdom is 8097)
