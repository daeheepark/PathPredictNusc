import cv2
import numpy as np
import visdom

def img2vis(vis : visdom.Visdom, img : np.ndarray, window_name = 'result'):
    vis.image(img.transpose(2,0,1), opt=dict(caption=window_name))

def drawtraj(img, output, num_modes = 2, trajnum = None , plot2vis = False, vis = None):
    trajs, probs = output[:,:-num_modes], output[:,-num_modes:]
    trajs, probs = np.array_split(trajs.detach().numpy(), num_modes, axis=1), np.array_split(probs.detach().numpy(), num_modes, axis=1) 
    
    if trajnum != None:
        trajs, probs = [trajs[trajnum]], [probs[trajnum]]

    for traj, prob in zip(trajs, probs):
        x, y = 400, 250
        cv2.circle(img, (int(y), int(x)), 2, (0,0,255), 2)
        for disp in np.array_split(traj, 12, axis=1):
            p1 = (int(y),int(x))
            x += disp[0][0] * 100
            y += disp[0][1] * 100
            p2 = (int(y),int(x))
            
            cv2.line(img, p1, p2, (255,0,255), 1 )
            cv2.circle(img, (int(y), int(x)), 2, (0,0,255), 2)
            
        cv2.putText(img, '%.2f' % prob, p2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 3)
        cv2.putText(img, '%.2f' % prob, p2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)

    if plot2vis == True:
        img2vis(vis, img, 'predicted trajectory')
    
    return 'str'

if __name__ == "__main__":
    from nuscenes import NuScenes
    from nuscenes.prediction import PredictHelper
    from nuscenes.prediction.input_representation.static_layers import StaticLayerRasterizer
    from nuscenes.prediction.input_representation.agents import AgentBoxesWithFadedHistory
    from nuscenes.prediction.input_representation.interface import InputRepresentation
    from nuscenes.prediction.input_representation.combinators import Rasterizer
    from nuscenes.prediction.models.backbone import ResNetBackbone
    from nuscenes.prediction.models.mtp import MTP
    import torch
    import matplotlib.pyplot as plt

    vis = visdom.Visdom()

    DATAROOT = '/home/dhpark/Data/nuscenes-v1.0/full_dataset/mini'
    nuscenes = NuScenes('v1.0-mini', dataroot=DATAROOT)
    helper = PredictHelper(nuscenes)
    static_layer_rasterizer = StaticLayerRasterizer(helper)
    agent_rasterizer = AgentBoxesWithFadedHistory(helper, seconds_of_history=1)
    mtp_input_representation = InputRepresentation(static_layer_rasterizer, agent_rasterizer, Rasterizer())

    instance_token_img, sample_token_img = 'bc38961ca0ac4b14ab90e547ba79fbb6', '7626dde27d604ac28a0240bdd54eba7a'
    anns = [ann for ann in nuscenes.sample_annotation if ann['instance_token'] == instance_token_img]
    img = mtp_input_representation.make_input_representation(instance_token_img, sample_token_img)


    

    backbone = ResNetBackbone('resnet50')
    num_modes = 2
    mtp = MTP(backbone, num_modes=num_modes)

    agent_state_vector = torch.Tensor([[helper.get_velocity_for_agent(instance_token_img, sample_token_img),
                                        helper.get_acceleration_for_agent(instance_token_img, sample_token_img),
                                        helper.get_heading_change_rate_for_agent(instance_token_img, sample_token_img)]])

    image_tensor = torch.Tensor(img).permute(2, 0, 1).unsqueeze(0)

    backbone_features = backbone(image_tensor)
    features = torch.cat([backbone_features, agent_state_vector], dim=1)

    output = mtp(image_tensor, agent_state_vector)

    drawtraj(img, output, num_modes=2, plot2vis=True, vis=vis)    

    pass