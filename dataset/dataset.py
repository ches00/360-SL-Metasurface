import torch 
import cv2 
import os
import numpy as np

from utils.render.openExr import read_exr_as_np


class CreateSyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, path, mode):
        self.path = path # 

        fisheye_path = os.path.join(self.path, "fisheye")
        pano_path = os.path.join(self.path, "pano")

        self.reflectance_path = os.path.join(fisheye_path, "Albedo")
        self.depth_path = os.path.join(fisheye_path, "Depth")
        self.normal_path = os.path.join(fisheye_path, "Normal")
        self.occlusion_path = os.path.join(fisheye_path, "Occlusion")
        
        self.gt_path = os.path.join(pano_path, "Depth/pano_cam/")
      
        self.file_list = os.listdir(os.path.join(self.reflectance_path, 'cam1'))
 
        if 'Thumbs.db' in self.file_list:
            self.file_list.remove('Thumbs.db')

        self.N = len(self.file_list)

    def __getitem__(self, i):
        
        ref_im_list = []
        depth_im_list = []
        occ_im_list = []
        normal_im_list = []

        idx = self.file_list[i][:-4]
        png_name = idx + '.png'
        exr_name = idx + '.exr'
        #png_name = '%d.png'%(idx)
        #exr_name = '%d.exr'%(idx)

        for idx in range(1, 3):

            cam_name = 'cam%d'%(idx)

            ref_im = cv2.imread(os.path.join(self.reflectance_path, cam_name, png_name))
            ref_im_torch = torch.from_numpy(ref_im).float() 
            ref_im_list.append(ref_im_torch)

            occ_im = cv2.imread(os.path.join(self.occlusion_path, cam_name, png_name))
            occ_im_torch = torch.from_numpy(occ_im)[..., 0].float() # 1 x H x W x 3      
            occ_im_list.append(occ_im_torch)

            depth_im = read_exr_as_np(os.path.join(self.depth_path, cam_name, exr_name))
            depth_torch = torch.from_numpy(depth_im)[..., 0].float()
            depth_im_list.append(depth_torch)

            normal_im = read_exr_as_np(os.path.join(self.normal_path, cam_name, exr_name))
            normal_torch = torch.from_numpy(normal_im).float()
            normal_torch[normal_torch.isnan()] = 0.
            normal_im_list.append(normal_torch)
            


        input_dict = {
            'ref_im_list': ref_im_list,
            'depth_im_list': depth_im_list,
            'occ_im_list': occ_im_list,
            'normal_im_list': normal_im_list, 
            #'gt': None, #gt_depth_torch,
            'name': png_name 
        }
        return input_dict


    def __len__(self):
        return self.N



class RealDataset(torch.utils.data.Dataset):
    def __init__(self, path, mode):
        self.path = path # "matching/ "

        self.cam3_list = os.listdir(os.path.join(self.path, 'cam3'))
        self.cam4_list = os.listdir(os.path.join(self.path, 'cam4'))
        self.cam3_list.sort() 
        self.cam4_list.sort()


        self.N = len(self.cam3_list)#file_list)

    def __getitem__(self, i):
        
        im_list = []


        for idx2 in range(1, 2):

            cam_name = 'cam%d'%(idx2)
            if idx2 == 3:
                png_name = self.cam3_list[i]
            else:
                png_name = self.cam4_list[i]
            img = cv2.resize(cv2.imread(os.path.join(self.path, cam_name, png_name)), (640, 400)) 
            img_torch = torch.from_numpy(img) / 255. #term
            im_list.append(img_torch ) 



        input_dict = {
            'img': im_list,
            'name': png_name[:-4] #idx
        }


        return input_dict


    def __len__(self):
        return self.N



    