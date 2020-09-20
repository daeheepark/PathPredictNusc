from nuscenes import NuScenes
import os
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
from PIL import Image

# This is the path where you stored your copy of the nuScenes dataset.
DATAROOT = '/home/dhpark/Data/nuscenes-v1.0/full_dataset/mini'
split = 'mini_train'
DATAPATH = os.path.join('./dataset', split)
dpathlist = ['image', 'state', 'traj']

nuscenes = NuScenes('v1.0-mini', dataroot=DATAROOT)

FUTURE_SEC = 6

from nuscenes.eval.prediction.splits import get_prediction_challenge_split
from nuscenes.prediction import PredictHelper

inst_samp_pair_list = get_prediction_challenge_split(split, dataroot=DATAROOT)
print(f"size of data : {len(inst_samp_pair_list):d}")
helper = PredictHelper(nuscenes)

dpathlist_ = []
for dlist in dpathlist:
    dpath = os.path.join(DATAPATH, dlist)
    dpathlist_.append(dpath)
    if not os.path.exists(dpath):
        os.makedirs(dpath)

from nuscenes.prediction.input_representation.static_layers import StaticLayerRasterizer
from nuscenes.prediction.input_representation.agents import AgentBoxesWithFadedHistory
from nuscenes.prediction.input_representation.interface import InputRepresentation
from nuscenes.prediction.input_representation.combinators import Rasterizer

static_layer_rasterizer = StaticLayerRasterizer(helper)
agent_rasterizer = AgentBoxesWithFadedHistory(helper, seconds_of_history=1)
mtp_input_representation = InputRepresentation(static_layer_rasterizer, agent_rasterizer, Rasterizer())

if os.path.isfile(os.path.join(DATAPATH, 'fn_list.txt')):
    os.remove(os.path.join(DATAPATH, 'fn_list.txt'))

fn_list = open(os.path.join(DATAPATH, 'fn_list.txt'), 'w')

for inst_samp_pair in tqdm(inst_samp_pair_list):
    fn_list.write(inst_samp_pair + '\n')
    instance_token, sample_token = inst_samp_pair.split("_")

    img = mtp_input_representation.make_input_representation(instance_token, sample_token)
    agent_state_vector = torch.Tensor([helper.get_velocity_for_agent(instance_token, sample_token),
                                    helper.get_acceleration_for_agent(instance_token, sample_token),
                                    helper.get_heading_change_rate_for_agent(instance_token, sample_token)])
    future_xy_local = torch.Tensor(helper.get_future_for_agent(instance_token, sample_token, seconds=FUTURE_SEC, in_agent_frame=True))

    Image.fromarray(img).save(os.path.join(dpathlist_[0], inst_samp_pair+'.jpg'))
    torch.save(agent_state_vector, os.path.join(dpathlist_[1], inst_samp_pair+'.state'))
    torch.save(future_xy_local, os.path.join(dpathlist_[2], inst_samp_pair+'.traj'))

fn_list.close()

# img = Image.open(os.path.join(dpathlist_[0], inst_samp_pair+'.jpg'))
img_tensor = transforms.ToTensor()(img)
print(type(img_tensor), img_tensor.shape)
print(type(agent_state_vector), agent_state_vector.shape)
print(type(future_xy_local), future_xy_local.shape)
