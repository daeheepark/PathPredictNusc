import torch
from torch.utils.data import DataLoader
import util 
import os

trainset_mini = util.DataSet('./dataset/mini_train', 'train')
valset_mini = util.DataSet('./dataset/mini_val', 'val')

train_mini_loader = DataLoader(trainset_mini, batch_size=32, shuffle=True, num_workers=0)
val_mini_loader = DataLoader(valset_mini, batch_size=32, shuffle=False, num_workers=0)

NUM_EPC = 200

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

import torch.optim as optim
import numpy as np

from nuscenes.prediction.models.backbone import ResNetBackbone
from nuscenes.prediction.models.mtp import MTP, MTPLoss
from tqdm import tqdm

num_modes = 2

backbone = ResNetBackbone('resnet18')
model = MTP(backbone, num_modes)
model = model.to(device)

loss_function = MTPLoss(num_modes, 1, 5)

current_ep_loss = 10000

optimizer = optim.SGD(model.parameters(), lr=0.1)

n_iter = 0

minimum_loss = 0

if os.path.isfile('./model/save_log.txt'):
    os.remove('./model/save_log.txt')

if num_modes == 2:

    # We expect to see 75% going_forward and
    # 25% going backward. So the minimum
    # classification loss is expected to be
    # 0.56234

    minimum_loss += 0.56234

for ep in range(NUM_EPC):
    loss_mean = []
    for img, agent_state_vector, ground_truth in tqdm(train_mini_loader):

        img[img != img] = 0
        agent_state_vector[agent_state_vector != agent_state_vector] = 0
        ground_truth[ground_truth != ground_truth] = 0

        img = img.to(device)
        agent_state_vector = agent_state_vector.to(device)
        ground_truth = ground_truth.to(device)

        optimizer.zero_grad()

        prediction = model(img, agent_state_vector)

        loss = loss_function(prediction, ground_truth.unsqueeze(1))

        # loss = loss_function(prediction, ground_truth)
        loss.backward()
        optimizer.step()

        current_loss = loss.cpu().detach().numpy()
        loss_mean.append(current_loss.item())

    ep_loss = np.mean(loss_mean)
    print(f"Current loss is {ep_loss:.4f} @ ep {n_iter:d}")
    # torch.save(model.state_dict(), './model/model_weight_best.pth')

    if ep_loss < current_ep_loss:
        print('lowest loss achieved')
        torch.save(model.state_dict(), './model/model_weight_best.pth')
        current_ep_loss = ep_loss
        f = open('./model/save_log.txt', 'a')
        f.write(f'loss {ep_loss:.3f} achieved at epoch {n_iter:d} \n')
        f.close()

    if np.allclose(ep_loss, minimum_loss, atol=1e-4):
        print(f"Achieved near-zero loss after {n_iter} iterations.")
        torch.save(model.state_dict(), f'./model/model_weight_{ep_loss:.2f}.pth')
        break

    n_iter += 1

torch.save(model.state_dict(), f'./model/model_weight_lastep.pth')