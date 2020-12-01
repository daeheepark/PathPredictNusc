import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

import os
from tqdm import tqdm
import numpy as np
import argparse
import json

from backbone import ResNetBackbone, MobileNetBackbone
from mtp import MTP, MTP_baseline, MTPLoss
# from mtp2 import MTP as MTP2
# from mtp2 import MTPLoss as MTPLoss2

from util import *

parser = argparse.ArgumentParser()
parser.add_argument('--name',       required=True,  type=str,   help='experiment name. saved to ./exps/[name]')
parser.add_argument('--max_epc',    default=30,    type=int)
parser.add_argument('--min_loss',   default=0.56234,type=float, help='minimum loss threshold that training stop')
parser.add_argument('--batch_size', default=32,     type=int)
parser.add_argument('--num_workers',default=4,      type=int)
parser.add_argument('--optimizer',  default='sgd', choices=['adam', 'sgd'])
parser.add_argument('--lr',         default=0.01, type=float)
parser.add_argument('--gpu_ids',    default='0,1',  type=str,   help='id of gpu ex) "0" or "0,1"')
parser.add_argument('--tsboard',    action='store_true',        help='use tensorboardX to viasuzliae experiments')
parser.add_argument('--diff',       action='store_true',        help='difference of trajectory as output')
parser.add_argument('--attention',  action='store_true',        help='transformer based attention model')

parser.add_argument('--num_modes',  default=2, type=int)
parser.add_argument('--backbone',   default='mobilenet_v2',     choices=['mobilenet_v2', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'])
parser.add_argument('--unfreeze',   default=0,      type=int,   help='number of layer of backbone CNN to update weight')

args = parser.parse_args('--name 1201_mode1_mbnet_1 --optimizer sgd --lr 0.1 --backbone mobilenet_v2 --num_modes 1 --max_epc 30 --batch_size 32'.split())
# args = parser.parse_args()

exp_path, train_path, val_path, infer_path, ckpt_path = make_path(args)

f = open(ckpt_path+'/'+'exp_config.txt', 'w')
json.dump(args.__dict__, f, indent=2)
f.close()

if os.path.isfile(ckpt_path+'/'+'save_log.txt'):
    os.remove(ckpt_path+'/'+'save_log.txt')

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

trainset = DataSet_proj('./dataset_chh/train', 'train')
valset = DataSet_proj('./dataset_chh/train_val', 'train_val')

train_loader = DataLoader(trainset, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers)
val_loader = DataLoader(valset, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers)

backbone = ResNetBackbone(args.backbone) if args.backbone.startswith('resnet') else MobileNetBackbone(args.backbone)

total_layer_ct = sum(1 for _ in backbone.parameters())
for i, param in enumerate(backbone.parameters()):
    if i < total_layer_ct - args.unfreeze:
        param.requires_grad = False
    else:
        param.requires_grad = True

if args.attention:
    model = MTP(backbone, is_diff=args.diff)
else:
    model = MTP_baseline(backbone, args.num_modes, is_diff=args.diff)
    
loss_function = MTPLoss(args.num_modes, 1, 5)
optimizer = optim.Adam(model.parameters(), lr=args.lr) if args.optimizer == 'adam' else optim.SGD(model.parameters(), lr=args.lr) 

torch.save(model, ckpt_path + '/' + 'model.archi')
torch.save(optimizer, ckpt_path + '/' + 'optim.archi')

model = nn.DataParallel(model)
model = model.to(device)

current_ep_loss = 10000

for epoch in range(args.max_epc):
    
    print('training start')
    model.train()
    loss_tr_mean = []

    for batch in tqdm(train_loader):
        raster, road, lane, agents, state, past, gt, diff = batch

        if not args.diff:
            gt_ = gt
        else:
            gt_ = diff

        raster, road, lane, agents, state, past, gt_ = NaN2Zero(raster), NaN2Zero(road), NaN2Zero(lane), NaN2Zero(agents), NaN2Zero(state), NaN2Zero(past),  NaN2Zero(gt_)
        raster, road, lane, agents, state, past, gt_ = raster.to(device), road.to(device), lane.to(device), agents.to(device), state.to(device), past.to(device), gt_.to(device)

        optimizer.zero_grad()

        if not args.attention:
            prediction = model(raster, state)
        else:
            prediction = model(road, lane, agents, state, past)

        loss = loss_function(prediction, gt_.unsqueeze(1))

        loss.backward()
        optimizer.step()

        loss_tr_mean.append(loss.item())

    print('validation start')
    model.eval()
    loss_val_mean = []
    for batch in tqdm(val_loader):
        raster, road, lane, agents, state, past, gt, diff = batch
        if not args.diff:
            gt_ = gt
        else:
            gt_ = diff

        raster, road, lane, agents, state, past, gt_ = NaN2Zero(raster), NaN2Zero(road), NaN2Zero(lane), NaN2Zero(agents), NaN2Zero(state), NaN2Zero(past),  NaN2Zero(gt_)
        raster, road, lane, agents, state, past, gt_ = raster.to(device), road.to(device), lane.to(device), agents.to(device), state.to(device), past.to(device), gt_.to(device)

        if not args.attention:
            prediction = model(raster, state)
        else:
            prediction = model(road, lane, agents, state, past)
        
        loss = loss_function(prediction, gt_.unsqueeze(1))

        loss_val_mean.append(loss.item())

    ep_loss_tr, ep_loss_val = np.mean(loss_tr_mean), np.mean(loss_val_mean)
    print(f"Current training loss is {ep_loss_tr:.4f} @ ep {epoch:d}")
    print(f"Current validation loss is {ep_loss_val:.4f} @ ep {epoch:d}")

    checkpoint = {'state_dict' : model.module.state_dict(), 'optimizer' : optimizer.state_dict(), 'loss' : ep_loss_tr, 'ep' : epoch}

    if ep_loss_val < current_ep_loss:
        print('best val loss achieved')
        torch.save(checkpoint, ckpt_path + '/' + 'weight_best.pth')
        current_ep_loss = ep_loss_val
        f = open(ckpt_path+'/'+'save_log.txt', 'a')
        f.write(f'\n loss {ep_loss_val:.3f} achieved at epoch {epoch:d}')
        f.close()

    if np.allclose(ep_loss_val, args.min_loss, atol=1e-4):
        print(f"Achieved loss under min_loss after {epoch} iterations.")
        torch.save(checkpoint, ckpt_path + '/' + f'weight_{ep_loss_val:.3f}.pth')
        break

torch.save(checkpoint, ckpt_path + '/' + 'weight_last.pth')