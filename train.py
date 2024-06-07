import torch 
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from utils.ArgParser import Argument
from dataset.dataset import CreateSyntheticDataset
from model.Metasurface import Metasurface
from utils.Camera import *
from Image_formation.renderer import *
from model.StereoMatching import DepthEstimator
import scipy.io


import matplotlib.pyplot as plt 
from torch.autograd import Variable

import GPUtil, os
from model.e2e import *


def grad_loss(output, gt):
    def one_grad(shift):
        ox = output[:, shift:] - output[:, :-shift]
        oy = output[:, :, shift:] - output[:, :, :-shift]
        gx = gt[:, shift:] - gt[:, :-shift]
        gy = gt[:, :, shift:] - gt[:, :, :-shift]
        loss = (ox - gx).abs().mean() + (oy - gy).abs().mean()
        return loss
    loss = (one_grad(1) + one_grad(2) + one_grad(3)) / 3.
    return loss


def illum_tv(output):
    def one_grad(shift):
        ox = output[shift:] - output[ :-shift]
        oy = output[ :, shift:] - output[ :, :-shift]

        loss = ox.abs().mean() + oy.abs().mean()
        return loss
    loss = (one_grad(1) + one_grad(2) + one_grad(3)) / 3.
    return loss


def train(opt, model, dataset_path):
    dataset_train = CreateSyntheticDataset(opt.train_path, 'train') # path 
    dataset_test = CreateSyntheticDataset(opt.valid_path, 'valid')
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=opt.batch_size, shuffle=True, num_workers=4)
    dataloader_valid = torch.utils.data.DataLoader(dataset_test, batch_size=opt.batch_size, shuffle=False, num_workers=4)

    meta_phase = torch.autograd.Variable(model.get_meta_phase(), requires_grad=True)
    optimizer_meta = torch.optim.Adam([meta_phase], lr=opt.lr)
    optimizer_net = torch.optim.Adam(list(model.parameters()), lr=opt.lr) 
    
    scheduler_meta = torch.optim.lr_scheduler.StepLR(optimizer_meta, step_size=200, gamma=0.2)
    scheduler_net = torch.optim.lr_scheduler.StepLR(optimizer_net, step_size=350, gamma=0.4)
    

    l1_loss = torch.nn.L1Loss() 
    l1_loss.requires_grad = True 

    device = torch.device(opt.device)
    model = model.to(device)

    writer = SummaryWriter(log_dir=opt.log)
    fisheye_mask = torch.from_numpy(np.load("./fisheye_mask.npy")).to(device)



    for epoch in range(1000):
        losses = []
        model.train()
        # minibatch
        for i, data in enumerate(dataloader_train):
            B = opt.batch_size
            
            ref_im_list = data['ref_im_list']
            depth_map_list = data['depth_im_list']
            occ_im_list = data['occ_im_list']
            normal_im_list = data['normal_im_list']

            # update meta-surface phase
            if optimizer_meta:
                model.update_phase(meta_phase)

            gt = 1.0 / (depth_map_list[0].to(device).float() * 10)     
            inv_depth_pred, synthetic_images = model(ref_im_list, depth_map_list, occ_im_list, normal_im_list)

            front_l1loss = l1_loss(gt[:, fisheye_mask], inv_depth_pred[0][:, fisheye_mask])
            front_tvloss = grad_loss(gt, inv_depth_pred[0])
            
            # pattern loss
            #pattern = model.get_pattern() 
            #illum = torch.nn.functional.grid_sample(pattern.repeat(1, 1, 1, 1), grid, mode='bilinear', padding_mode='zeros').squeeze(0).squeeze(0)
            #illum_loss = 1 / illum_tv(illum / illum.max())
            
            
            loss = front_l1loss + front_tvloss * 0.4 # + 0.01 * illum_loss 
            print("{0}th iter : {1}".format(i, loss.item()))
            losses.append(loss.item())


            if optimizer_meta :
                optimizer_meta.zero_grad()
            if optimizer_net :
                optimizer_net.zero_grad()
            loss.backward()
            if optimizer_meta :
                optimizer_meta.step()
            if optimizer_net :
                optimizer_net.step()


        print("[{0}/1000 epoch - Train loss : {1}".format(epoch, sum(losses)/len(losses)))


        # Test        
        model.eval() 
        losses = []
        with torch.no_grad():
            for j, data in enumerate(dataloader_valid):
                B = opt.batch_size
            
                ref_im_list = data['ref_im_list']
                depth_map_list = data['depth_im_list']
                occ_im_list = data['occ_im_list']
                normal_im_list = data['normal_im_list']

                gt = 1.0 / (depth_map_list[0].to(device).float() * 10)
                inv_depth_pred, _ =  model(ref_im_list, depth_map_list, occ_im_list, normal_im_list)
                
                front_l1loss = l1_loss(gt[:, fisheye_mask], inv_depth_pred[0][:, fisheye_mask])
                front_tvloss = grad_loss(gt, inv_depth_pred[0])
                

                
                loss = front_l1loss + front_tvloss * 0.4
                losses.append(loss.item())

            print("[{0}/1000 epoch - validation loss : {1}".format(epoch, sum(losses)/len(losses)))
            


        
        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(opt.log, "model_epoch_%d.pth"%(epoch)))
            np.save(os.path.join(opt.log, "phase_epoch_%d.npy"%(epoch)), meta_phase.detach().cpu().numpy())









if __name__ == "__main__":
    parser = Argument()
    args = parser.parse()

    device = torch.device(args.device)

    metasurface = Metasurface(args, device)


    radian_90 = math.radians(90)


    cam1 = FisheyeCam(args, (0.05, 0.05, 0), (radian_90, 0, 0), 'cam1', device, args.cam_config_path)
    cam2 = FisheyeCam(args, (-0.05, 0.05, 0), (radian_90, 0, 0), 'cam2', device, args.cam_config_path)

    
    # Front-back / in training time, we just trained cam1-cam2 front system.
    cam_calib = [cam1, cam2]

    renderer = ActiveStereoRenderer(args, metasurface, cam_calib, device)
    pano_cam = PanoramaCam(args, (0, 0, 0), (radian_90, 0, 0), 'pano', device)
    estimator = DepthEstimator(pano_cam, cam_calib, device, args)

    e2e_model = E2E(metasurface, renderer, estimator)

    train(args, e2e_model, args.input_path)

