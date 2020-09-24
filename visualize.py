import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.transforms as transforms

import numpy as np
import os
import argparse
import visdom

import util

def unnormalize(img):
    mean = torch.tensor([1, 2, 3], dtype=torch.float32)
    std = torch.tensor([2, 2, 2], dtype=torch.float32)
    transforms

vis = visdom.Visdom()

parser = argparse.ArgumentParser()
parser.add_argument('--name',       required=True,  type=str)
parser.add_argument('--ep',         required=True,  type=str)
parser.add_argument('--whichset',   default='val',  type=str,   choices=['val', 'train_val'], help='dataset to visualize')
parser.add_argument('--batch_id',   default=0,      type=int,   help='batch id to visualize')
parser.add_argument('--batch_size', default=100,    type=int,   help='number of image to show')
parser.add_argument('--num_workers',default=8,      type=int)
parser.add_argument('--gpu_ids',    default='2',    type=str,   help='id of gpu ex) "0" or "0,1"')
parser.add_argument('--shuffle',    action='store_true')
parser.add_argument('--savepic',    action='store_true')

args = parser.parse_args('--name test --ep best --whichset train_val --batch_id 0 --shuffle'.split())
# args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

dataset = util.DataSet('./dataset/' + args.whichset, args.whichset)
dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=args.shuffle)

exp_path, train_path, val_path, infer_path, ckpt_path = util.make_path(args)

model = torch.load(ckpt_path + '/' + 'model.archi').to(device)
# model = nn.DataParallel(model).to(device)
model.load_state_dict(torch.load(ckpt_path + '/' + 'weight_' + args.ep + '.pth')['state_dict'])

dataiter = iter(dataloader)
for _ in range(args.batch_id + 1):
    data2show = next(dataiter)

img, state, gt = data2show
# img, state, gt = util.NaN2Zero(img), util.NaN2Zero(state), util.NaN2Zero(gt)
img, state, gt = img.to(device), state.to(device), gt.to(device)

prediction = model(img, state)

for idx, (img_, state_, gt_, pred_) in enumerate(zip(img, state, gt, prediction)):
    img_, state_, gt_, pred_ = img_.cpu(), state_.cpu(), gt_.cpu(), pred_.detach().cpu()
    pred_ = util.NaN2Zero(pred_)
    # img_np = np.asarray(img_)*255, dtype=np.uint8).transpose(1,2,0)
    # img_np = (img_*255).numpy().astype('uint8').transpose(1,2,0)
    img_orig = util.restore_img(img_)
    img_pil = transforms.ToPILImage()(img_orig)
    img_resize = transforms.Resize(500)(img_pil)

    result_img = util.visualize(np.array(img_resize), num_modes=2, prediction=pred_, gt=gt_)#, traj_slice=(1,2))
    vis.image(result_img.transpose(2,0,1), opts=dict(title=args.name + ' ep: ' + args.ep + ' batch id: ' + str(args.batch_id) + ' batch idx: ' + str(idx)))
    # vis.image(img_orig, opts=dict(title=args.name + ' batch id: ' + str(args.batch_id) + '_' + str(idx)))
