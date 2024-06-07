import torch
import torch.nn.functional as F
import numpy as np 
import cv2
import os, math

from utils.Camera import FisheyeCam
from utils.ArgParser import Argument 
from utils.render.noise import GaussianNoise
from utils.render.openExr import read_exr_as_np
from model.Metasurface import Metasurface



def img_show(img):
	img_np = (img).cpu().numpy()

	cv2.imshow('test', img_np)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


def dir_to_sin(pts, num_cells):
	### pts : B x 3 x N 
	x = pts[:, 0, :]
	y = pts[:, 1, :]
	z = pts[:, 2, :]


	sin_phi = x / torch.sqrt(x**2 + y**2) # N [-1, 1]
	sin_theta = -z / torch.norm(pts, dim=1) # N

	u = (sin_phi + 1) * (num_cells/2)
	v = (sin_theta +1) * (num_cells/2)

	return (u, v)



class ActiveStereoRenderer:
	def __init__(self, opt, metasurface, fisheye_cams, device):

		self.opt = opt 
		self.device = device
		
		self.metasurface = metasurface 
		self.cam_calib = fisheye_cams

		self.resolution_x = opt.fisheye_resolution_x
		self.resolution_y = opt.fisheye_resolution_y


		self.ambient_light_off = opt.ambient_light_off # True or False
		self.noise = GaussianNoise(0, opt.noise_gaussian_stddev, self.device)

	def find_pattern(self, pts, pattern):
		### pts : B x (3xN) 
		### pattern : n_phase x n_phase 

		x = -pts[:, 0, :]
		y = pts[:, 2, :]
		z = pts[:, 1, :]

		B = x.shape[0]

		n_phase = self.opt.N_phase 
		wvl = self.opt.wave_length
		p = self.opt.pixel_pitch 

		norm = torch.sqrt(x**2 + y**2 + z**2)

		fx = (x/norm) / wvl * p 
		fy = (y/norm) / wvl * p

		dist_factor = 1/(norm**2)

		H, W = self.resolution_y, self.resolution_x 

		x_base = fx.reshape(B, H, W)
		y_base = fy.reshape(B, H, W)
		grid = torch.stack((x_base, y_base), dim=-1) * 2 # B x W x H x 2 

		output = F.grid_sample(pattern.repeat(B, 1, 1, 1), grid, mode='bilinear', padding_mode='zeros') 

		return output[:, 0, ...] * dist_factor.reshape(B, H, W) # B x H x W 

		

	def find_pattern_sin(self, pts, pattern):
		resolution = self.opt.N_phase * self.opt.N_supercell * 0.5
		u, v = dir_to_sin(pts, resolution)

		pattern_flatten = pattern.flatten().float()

		# grid_sample test
		B = pts.shape[0]
		H, W = self.resolution_y, self.resolution_x
		x_base = u.reshape(B, H, W) / resolution
		y_base = v.reshape(B, H, W) / resolution 

		grid = torch.stack((x_base, y_base), dim=3)

		output = F.grid_sample(pattern.repeat(B, 1, 1, 1), 2*grid-1, mode='bilinear', padding_mode='zeros')
		result = output[:, 0, ...] # B x H x W 

		return result







	def render(self, ref_im_list, depth_map_list, occ_list, normal_list, pattern_fixed=None):
	#---------------------------SCENE SETUP-----------------------
	#                            scene
	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	#     cam_1                MetaSurface          cam_2
	#                          light source                
	#     cam_3                                     cam_4
	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	#                            scene
		num_cameras = len(self.cam_calib)
		result = []

		# make panorama illumination image 
		pattern_360 = self.metasurface.propagate()**2
		if pattern_360.shape[0] > 500:
			pattern_360 = torch.nn.functional.interpolate(pattern_360.unsqueeze(0).unsqueeze(0), size=(500, 500), mode='bilinear')[0][0]
   

		for i in range(2):

			cam = self.cam_calib[i]


			ref_im = ref_im_list[i].to(self.device)/255. 
			depth_map = depth_map_list[i].to(self.device)
			occ = occ_list[i].to(self.device)/255.
			normal = normal_list[i].to(self.device)

			B = ref_im.shape[0] # batch size

			ambient_power = torch.tensor(np.random.uniform(low=self.opt.ambient_power_min, high=self.opt.ambient_power_max, size=(B, 1, 1)).astype(np.float32)).to(self.device)
			laser_power =  8e-8 #1e-7 - ours #5e-8 #tv2 : 8e-8#3e-8#0.8 torch.tensor(np.random.uniform(low=self.opt.laser_power_min, high=self.opt.laser_power_max, size=(B, 1, 1)).astype(np.float32), device=self.opt.device)
			
			xyz_pts_norm = cam.get_whole_pts().unsqueeze(0) # 1 x 3 x N
			xyz_pts = xyz_pts_norm * depth_map.reshape(B, -1).unsqueeze(1) * 10 

			w = torch.ones(B, 1, xyz_pts.shape[2]).to(self.device) # B x 1 x N 
			homo_coord_xyz = torch.cat([xyz_pts, w], dim=1) # B x 4 x N
			xyz_pts_o = torch.matmul(cam.get_extrinsic(), homo_coord_xyz.permute(1, 2, 0).reshape(4, -1)) # 
			
			xyz_pts_o = (xyz_pts_o / xyz_pts_o[-1])[:3].reshape(3, -1, B) #3 x N x B
			xyz_pts_o = xyz_pts_o 

			# Far-field propagation
			pattern = self.find_pattern(xyz_pts_o.permute(2, 0, 1), pattern_360).float()
			pattern_img = occ * (laser_power * pattern)
			pattern_occ = pattern_img + ambient_power # + 0.04

			R_img = ref_im[..., 0] * (ambient_power) # * occ)
			G_img = ref_im[..., 1] *  pattern_occ  + 0.2 # Green laser 
			B_img = ref_im[..., 2] * (ambient_power) # * occ)
			
			im_sim = torch.stack([R_img, G_img, B_img], axis=-1)


			normal_origin = normal.reshape(B, -1, 3) * 2 - 1  # B x (w x H) x 3
			dot_result = normal_origin * -xyz_pts_o.permute(2, 1, 0) # B x (w x H) x 3
			normal_size = torch.norm(normal_origin, dim=2)
			xyz_size = torch.norm(dot_result, dim=2)
			normal_size[normal_size==0] = 1
			xyz_size[xyz_size==0] = 1
			cos_theta = dot_result.sum(dim=2) / (normal_size * xyz_size)

			im_sim_normal = im_sim * cos_theta.reshape(im_sim.shape[:-1]).unsqueeze(-1) # (B, H, W, 3) * (B, H, W, 1 )

			
			noise = self.noise.sample(ref_im.shape)
			im_sim_noisy = im_sim_normal + noise 

			#sensor clamping 
			im_sim_noisy_clamped = torch.clamp(im_sim_noisy, min=0, max=1)
			result.append(im_sim_noisy_clamped)
			
		return result, None#illum_img
	
	



