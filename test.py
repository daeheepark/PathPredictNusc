import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import visdom
import os
import util
import matplotlib.pyplot as plt

from nuscenes.prediction.models.backbone import ResNetBackbone
from nuscenes.prediction.models.mtp import MTP, MTPLoss


vis = visdom.Visdom()

DATAROOT = '/home/dhpark/Projects/PathPredictNusc/dataset/mini_train/'
datalist = ['image', 'state', 'traj']
resultdir = './result/trainmini_mode(2)_ep(200)'
SAVE_RESULT = False

if SAVE_RESULT is True:
    if os.path.isdir(resultdir) is False:
        os.makedirs(resultdir)

fn_list = open(os.path.join(DATAROOT, 'fn_list.txt'), 'r')

fns = []
while True:
    line = fn_list.readline()
    if not line:
        fn_list.close()
        break
    fns.append(line.strip())

idx_list = np.arange(1,len(fns),20)

num_modes = 2

backbone = ResNetBackbone('resnet18')
model = MTP(backbone, num_modes)

model.load_state_dict(torch.load('./model/model_weight_best.pth'))

for idx in idx_list:
    img = Image.open(os.path.join(DATAROOT, datalist[0], fns[idx]+'.jpg'))
    state = torch.load(os.path.join(DATAROOT, datalist[1], fns[idx]+'.state'))
    gt = torch.load(os.path.join(DATAROOT, datalist[2], fns[idx]+'.traj'))


    # img_vis = util.visualize(np.asarray(img), num_modes=2, gt=gt)

    # plt.figure(0)
    # plt.imshow(img_vis)
    # plt.show()

    # img_vis = img_vis.transpose(2, 0, 1)
    # print(img_vis.shape)
    # vis.image(img_vis, 'test')





    img_test = transforms.ToTensor()(img).unsqueeze(0)
    state_test = state.unsqueeze(0)
    gt_test = gt.unsqueeze(0)

    prediction = model(img_test, state_test)
    # print(prediction)

    # img[img != img] = 0
    prediction[prediction != prediction] = 0
    gt_test[gt_test != gt_test] = 0

    img_visualize = util.visualize(np.asarray(img), num_modes=2, prediction=prediction, gt=gt_test, traj_slice=(1,2))

    vis.image(img_visualize.transpose(2,0,1), str(idx))

    if SAVE_RESULT is True:
        Image.fromarray(img_visualize).save(os.path.join(resultdir, f'{fns[idx]:s}.jpg'))

# print(img_visualize)
# plt.imshow(img_visualize)
# plt.show()


