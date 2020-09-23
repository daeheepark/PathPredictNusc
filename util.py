import numpy as np
import cv2
from PIL import Image
import os
import os.path as osp
import torch
import torchvision.transforms as transforms

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
    if prediction is not None:
        trajs, probs = prediction[:-num_modes], prediction[-num_modes:]
        trajs, probs = np.array_split(trajs.unsqueeze(0).numpy(), num_modes, axis=1), np.array_split(probs.unsqueeze(0).numpy(), num_modes, axis=1) 
    
        if traj_slice is not None:
            trajs, probs = trajs[traj_slice[0]: traj_slice[1]], probs[traj_slice[0]: traj_slice[1]] 

        for traj, prob in zip(trajs, probs):
            x, y = 400, 250
            cv2.circle(img, (int(y), int(x)), 2, (0,0,255), 2)
            for disp in np.array_split(traj, 12, axis=1):
                x_cv, y_cv = y+int(disp[0][0])*10, x-int(disp[0][1])*10
                cv2.circle(img, (x_cv, y_cv), 2, (0,0,255), 2)
            #     p1 = (int(y),int(x))
            #     x -= disp[0][1] * 10
            #     y -= disp[0][0] * 10s
            #     p2 = (int(y),int(x))

            #     cv2.line(img, p1, p2, (255,0,255), 1 )
            #     cv2.circle(img, (int(y), int(x)), 2, (0,0,255), 2)

            # cv2.putText(img, '%.2f' % prob, p2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 3)
            # cv2.putText(img, '%.2f' % prob, p2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
                
            # cv2.putText(img, '%.2f' % prob, p2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 3)
            cv2.rectangle(img, (x_cv+5,y_cv-20), (x_cv+50,y_cv), (255,255,255), -1)
            cv2.putText(img, '%.2f' % prob, (x_cv+8,y_cv-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    
    if gt is not None:
        traj = np.asarray(gt).reshape(-1,24)
        x, y = 400, 250
        cv2.circle(img, (int(y), int(x)), 2, (0,255,0), 2)
        for disp in np.array_split(traj, 12, axis=1):
            x_cv, y_cv = y+int(disp[0][0])*10, x-int(disp[0][1])*10
            cv2.circle(img, (x_cv, y_cv), 2, (0,255,0), 2)

        # cv2.putText(img, 'gt', p2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 3)
        cv2.rectangle(img, (x_cv+5,y_cv-20), (x_cv+30,y_cv), (255,255,255), -1)
        cv2.putText(img, 'gt', (x_cv+8,y_cv-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        
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
