import numpy as np
import cv2
from PIL import Image
import os
import os.path as osp
import torch
import torchvision.transforms as transforms

TRAJ_COLORS = [(0,255,255), (255,128,0), (255,0,255), (0,0,255)]

class DataSet_differential(torch.utils.data.Dataset):
    def __init__(self, dataroot, split=None): # dataroot should contatin /image, /state, /traj in it.
        if not os.path.isdir(dataroot):
            raise NameError('dataroot does not exist')

        fn_list = open(os.path.join(dataroot, 'fn_list.txt'), 'r')

        fns = []
        while True:
            line = fn_list.readline()
            if not line:
                fn_list.close()
                break
            fns.append(line.strip())

        if split not in ['train', 'train_val', 'val']:
            raise NameError('split should be "train" / "trai_val" / "val"')

        self.dataroot = dataroot
        self.fns = fns
        self.split = split
        transform = transforms.Compose([transforms.Resize(224),transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
        self.transform = transform                        
    
    def __getitem__(self, index):
        fn = self.fns[index]
        image = Image.open(os.path.join(self.dataroot, 'image', fn+'.jpg'))
        image = self.transform(image)
        agent_state_vector = torch.load(os.path.join(self.dataroot, 'state', fn+'.state'))
        
        ground_truth = torch.load(os.path.join(self.dataroot, 'traj', fn+'.traj'))
        diff = ground_truth[1:] - ground_truth[:-1]

        return image, agent_state_vector, ground_truth, diff
        
    def __len__(self):
        return len(self.fns)

def restore_img(img):
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    
    img = img * torch.tensor(std).view(3, 1, 1) + torch.tensor(mean).view(3, 1, 1)
    return img

def make_path(args):
    exp_path = osp.join('./exps', args.name)
    train_path = osp.join(exp_path, './train')
    val_path = osp.join(exp_path, './val')
    ckpt_path = osp.join(exp_path, './ckpt')
    infer_path = osp.join(exp_path, './infer')
    
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
        os.makedirs(train_path)
        os.makedirs(val_path)
        os.makedirs(ckpt_path)
        os.makedirs(infer_path)
        print(exp_path + ' is built.')
    else:
        print(exp_path + ' already exsits.')

    return exp_path, train_path, val_path, infer_path,ckpt_path

def NaN2Zero(target : torch.Tensor):
    target[target != target] = 0
    return target

def visualize(img : np.ndarray, num_modes = 2, prediction = None, gt = None, traj_slice : tuple((int, int)) = None) :
    x, y = 400, 250

    if gt is not None:
        for gt_ in gt:
            x_cv, y_cv = int(y+gt_[0].item()*10), int(x-gt_[1].item()*10)
            cv2.drawMarker(img, (x_cv, y_cv), (0,255,0), cv2.MARKER_SQUARE, 6, 4, cv2.FILLED)

        size, _ = cv2.getTextSize('GT', cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        sx, sy = size
        cv2.rectangle(img, (x_cv,y_cv-sy-6), (x_cv+sx+6,y_cv), (255,255,255), -1)
        cv2.putText(img, 'GT', (x_cv+3,y_cv-3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

    if prediction is not None:
        trajs, probs = prediction[:-num_modes], prediction[-num_modes:]
        trajs, probs = np.array_split(trajs.numpy(), num_modes, axis=0), np.array_split(probs.numpy(), num_modes, axis=0) 
    
        if traj_slice is not None:
            trajs, probs = trajs[traj_slice[0]: traj_slice[1]], probs[traj_slice[0]: traj_slice[1]] 

        for i, (traj, prob) in enumerate(zip(trajs, probs)):
            traj_x, traj_y = traj[0::2], traj[1::2]
            prob_ = prob[0]
            prev_x, prev_y = y, x
            for traj_x_, traj_y_ in zip(traj_x, traj_y):
                x_cv, y_cv = int(y+traj_x_*10), int(x-traj_y_*10)
                cv2.drawMarker(img, (x_cv, y_cv), TRAJ_COLORS[i], cv2.MARKER_CROSS, 6, 3, cv2.FILLED)
                cv2.arrowedLine(img, (prev_x, prev_y), (x_cv, y_cv), TRAJ_COLORS[i], 1, cv2.LINE_AA, tipLength=0.1)
                prev_x, prev_y = x_cv, y_cv

            prob_ = f'{prob[0]:.2f}'
            size, _ = cv2.getTextSize(prob_, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            sx, sy = size
            cv2.rectangle(img, (x_cv,y_cv-sy-6), (x_cv+sx+6,y_cv), (255,255,255), -1)
            cv2.putText(img, prob_, (x_cv+3,y_cv-3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    
    return img

class DataSet(torch.utils.data.Dataset):
    def __init__(self, dataroot, split=None): # dataroot should contatin /image, /state, /traj in it.
        if not os.path.isdir(dataroot):
            raise NameError('dataroot does not exist')

        fn_list = open(os.path.join(dataroot, 'fn_list.txt'), 'r')

        fns = []
        while True:
            line = fn_list.readline()
            if not line:
                fn_list.close()
                break
            fns.append(line.strip())

        if split not in ['train', 'train_val', 'val']:
            raise NameError('split should be "train" / "trai_val" / "val"')

        self.dataroot = dataroot
        self.fns = fns
        self.split = split
        transform = transforms.Compose([transforms.Resize(224),transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
        self.transform = transform                        
    
    def __getitem__(self, index):
        fn = self.fns[index]
        image = Image.open(os.path.join(self.dataroot, 'image', fn+'.jpg'))
        image = self.transform(image)
        agent_state_vector = torch.load(os.path.join(self.dataroot, 'state', fn+'.state'))
        
        ground_truth = torch.load(os.path.join(self.dataroot, 'traj', fn+'.traj'))

        return image, agent_state_vector, ground_truth
        
    def __len__(self):
        return len(self.fns)
