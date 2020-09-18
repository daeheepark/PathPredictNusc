from nuscenes import NuScenes

# This is the path where you stored your copy of the nuScenes dataset.
DATAROOT = '/home/dhpark/Data/nuscenes-v1.0/full_dataset/trainval/v1.0-trainval_meta'
DATAROOT_map = '/home/dhpark/Data/nuscenes-v1.0/full_dataset/mini'

nuscenes = NuScenes('v1.0-trainval', dataroot=DATAROOT)

from nuscenes.eval.prediction.splits import get_prediction_challenge_split
train = get_prediction_challenge_split("train", dataroot=DATAROOT)
train[:5]
print(type(train))