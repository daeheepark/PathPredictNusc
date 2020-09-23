import torch
import torchvision.transforms as transforms
import torch.nn as nn
from PIL import Image
import numpy as np
import visdom
import os
import util
import matplotlib.pyplot as plt

from nuscenes.prediction.models.backbone import ResNetBackbone, MobileNetBackbone
from nuscenes.prediction.models.mtp import MTP, MTPLoss

transform = transforms.Compose([transforms.Resize(224),transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

vis = visdom.Visdom()

DATAROOT = './dataset/val/'
datalist = ['image', 'state', 'traj']
resultdir = './exps/test/ckpt/weight.best.pth'
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

idx_list = np.arange(1,len(fns),30)

num_modes = 2

backbone = ResNetBackbone('resnet18')
# backbone = MobileNetBackbone('mobilenet_v2')
model = MTP(backbone, num_modes)
# model = nn.DataParallel(model)
model = model.to(device)

# model.load_state_dict(torch.load('./model/model_weight_best.pth'))

for idx in idx_list:
    img = Image.open(os.path.join(DATAROOT, datalist[0], fns[idx]+'.jpg'))
    state = torch.load(os.path.join(DATAROOT, datalist[1], fns[idx]+'.state'))
    gt = torch.load(os.path.join(DATAROOT, datalist[2], fns[idx]+'.traj'))

    img_test = transform(img).unsqueeze(0)
    state_test = state.unsqueeze(0)
    gt_test = gt.unsqueeze(0)

    img_test, state_test, gt_test = img_test.to(device), state_test.to(device), gt_test.to(device)

    prediction = model(img_test, state_test)
    print(prediction)

    # img[img != img] = 0
    prediction[prediction != prediction] = 0
    gt_test[gt_test != gt_test] = 0
    

    img_visualize = util.visualize(np.asarray(img), num_modes=2, prediction=prediction.squeeze(0).detach().cpu(), gt=gt_test.cpu())#, traj_slice=(1,2))

    vis.image(img_visualize.transpose(2,0,1), opts=dict(title=str(idx)))

    if SAVE_RESULT is True:
        Image.fromarray(img_visualize).save(os.path.join(resultdir, f'{fns[idx]:s}.jpg'))

