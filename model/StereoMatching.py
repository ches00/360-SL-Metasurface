import torch, gc
import torch.nn as nn
from utils.Camera import *
from utils.ArgParser import Argument
from utils.render.openExr import read_exr_as_np

import matplotlib.pyplot as plt
from utils.render.openExr import read_exr_as_np
import torchvision.transforms as T

from utils.net.basic_layer import convbn, convbn_3d
import torch.nn.functional as F




class EdgeAwareRefinement(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        self.edge_in = nn.Sequential(
            convbn(4, 32, kernel_size=3, stride=1, pad=1, dilation=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        
        self.edge_filter = nn.ModuleList() 
        for i in range(5):
            self.edge_filter.append(
                BasicBlock(
                    32, 32, stride=1, downsample=None, pad=1, dilation=1
                )
            )
        
        self.edge_out = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)



        self.conv2d_feature = nn.Sequential(
            convbn(2, 32, kernel_size=3, stride=1, pad=1, dilation=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

		

        self.residual_astrous_blocks = nn.ModuleList()
        astrous_list = [1, 2, 3, 4, 3, 2, 1]
        for di in astrous_list:
            self.residual_astrous_blocks.append(
                BasicBlock(
                    32, 32, stride=1, downsample=None, pad=1, dilation=di))

        self.conv2d_out = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, low_disparity, corresponding_rgb):
        #output = torch.unsqueeze(low_disparity, dim=1)
        twice_disparity = F.interpolate(
            low_disparity,
            size=corresponding_rgb.size()[-2:],
            mode='bilinear',
            align_corners=False)

        edge_input = self.edge_in(torch.cat([corresponding_rgb, twice_disparity], dim=1))

        for filter in self.edge_filter:
            edge_input = filter(edge_input)

        edge_map = self.edge_out(edge_input)

    
        output = self.conv2d_feature(
            torch.cat([twice_disparity, edge_map], dim=1))

        for astrous_block in self.residual_astrous_blocks:
            output = astrous_block(output)

        return nn.ReLU(inplace=True)(torch.squeeze(
            twice_disparity + self.conv2d_out(output) * 0.01, dim=1))


class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride, downsample, pad, dilation):
        super().__init__()
        self.conv1 = nn.Sequential(
            convbn(in_channel, out_channel, 3, stride, pad, dilation),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)

        if self.downsample is not None:
            x = self.downsample(x)

        out = x + out
        return out

class FeatureExtractor(nn.Module):
	def __init__(self, k, in_channel=3, out_channel=32):
		super().__init__()
		self.k = k 
		self.downsample = nn.ModuleList()
		self.in_channel = in_channel
		self.out_channel = out_channel 

		for _ in range(k):
			self.downsample.append(
				nn.Conv2d(
					self.in_channel,
					self.out_channel,
					kernel_size=5,
					stride=2,
					padding=2
				)
			)
			self.in_channel = self.out_channel 

		self.residual_blocks = nn.ModuleList() 
		for _ in range(6):
			self.residual_blocks.append(
				BasicBlock(self.out_channel, self.out_channel, stride=1, downsample=None, pad=1, dilation=1)
			)
		self.conv_alone = nn.Conv2d(self.out_channel, self.out_channel, kernel_size=3, stride=1, padding=1)

	def forward(self, img):
		output = img 
		for d in self.downsample:
			output = d(output)

		for block in self.residual_blocks:
			output = block(output)

		return self.conv_alone(output)


class DepthEstimator(nn.Module):
	def __init__(self, pano_cam, fish_cam, device, opt) :
		super().__init__()
		self.pano_cam = pano_cam 
		self.fisheye_cams = fish_cam 

		self.r_min = opt.min_depth 
		self.r_max = opt.max_depth 
		self.count_candidate = opt.N_depth_candidate #

		self.device = device

		self.fisheye_resolution_x = opt.fisheye_resolution_x
		self.fisheye_resolution_y = opt.fisheye_resolution_y

		self.pano_resolution_x = opt.pano_resolution_x 
		self.pano_resolution_y = opt.pano_resolution_y 

		self.num_features = 34  
		

		self.filter = nn.ModuleList()
		for _ in range(7):
			self.filter.append(
				nn.Sequential(
					 convbn_3d(self.num_features, self.num_features, kernel_size=3, stride=1, pad=1), 
					 nn.LeakyReLU(negative_slope=0.2, inplace=True)
				)
			)

		self.conv3d_alone = nn.Conv3d(self.num_features, 1, kernel_size=3, stride=1, padding=1)

		self.feature_extractor = FeatureExtractor(2, in_channel=3, out_channel=self.num_features)
		self.edge_aware_refinements = EdgeAwareRefinement(4) 






	# test reference spherical sweeping
	def forward(self, img_list):
		self.device = img_list[0].device 
		feature_list = [] 
		R = self.count_candidate
		B = img_list[0].shape[0]

		r_range = 1/torch.linspace(1/self.r_max, 1/self.r_min, R).to(self.device).flip(0) 
		high_resolution_pred = []
		for img in img_list:
			feature =  self.feature_extractor(img.permute(0, -1, 1, 2) ** 1/(2.2))
			feature_list.append(feature)

		for i, j in [(0, 1)]: #, (1, 0)]
			ref_cam = self.fisheye_cams[i]
			target_cam = self.fisheye_cams[j]

			norm_pts = ref_cam.get_whole_pts().unsqueeze(-1) 

			sweeping_pts = norm_pts *r_range # 3 x N x R 
			sweeping_pts = sweeping_pts.reshape(3, -1)
			w = torch.ones(sweeping_pts.shape[1]).unsqueeze(0).to(self.device)
			pts = torch.cat([sweeping_pts, w])

			world_pts = torch.matmul(ref_cam.get_extrinsic(), pts)
			world_pts = (world_pts/world_pts[-1])
			world_pts = torch.matmul(torch.inverse(target_cam.get_extrinsic()),  world_pts)#torch.inverse(target_cam.get_extrinsic()), pts)
			world_pts = (world_pts/world_pts[-1])[:3]

			uv = target_cam.world2pixel(world_pts.reshape(3, -1))

			u = uv[0]
			v = uv[1]

			H, W = self.fisheye_resolution_y, self.fisheye_resolution_x
			x_base = u.reshape(1, H, W, -1) / self.fisheye_resolution_x
			x_base = x_base.permute(0, -1, 1, 2) # B x D x H x W
			y_base = v.reshape(1, H, W, -1) / self.fisheye_resolution_y
			y_base = y_base.permute(0, -1, 1, 2) # B x D x H x W
			z_base = torch.zeros_like(x_base)

			grid = torch.stack((x_base, y_base, z_base), dim=4).repeat(B, 1, 1, 1, 1) # 1 x D x H x W x 3


			target_img = feature_list[j].unsqueeze(-1)
			target_img = target_img.permute(0, 1, -1, 2, 3)

			intensity = torch.nn.functional.grid_sample(target_img, 2*grid-1, mode='bilinear', padding_mode='zeros') # B x C x R x H x W	
			intensity = torch.nn.functional.interpolate(intensity.reshape(B, -1, H, W), size=(H//4, W//4), mode='bilinear').reshape(B, -1, R, H//4, W//4)

			src_img = feature_list[i].unsqueeze(2) 

			cost_volume = torch.abs(src_img - intensity) # B x C x R x H x W

			for f in self.filter:
				cost_volume = f(cost_volume)

			# softmax
			cost_volume_filtered = self.conv3d_alone(cost_volume).squeeze(1)

			prob = torch.nn.functional.softmax(cost_volume_filtered, dim=1)
			expectation = prob * r_range.reshape(1, R, 1, 1)
			inv_depth_pred = 1.0 / expectation.sum(dim=1).unsqueeze(1) 
			result = self.edge_aware_refinements(inv_depth_pred, img_list[i].permute(0, -1, 1, 2))
			high_resolution_pred.append(result)

		return high_resolution_pred

