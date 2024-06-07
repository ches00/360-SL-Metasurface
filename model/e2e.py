import torch
import torch.nn as nn 
from torch.autograd import Variable



class E2E(nn.Module):
    def __init__(self, metasurface, renderer, estimator):
        super().__init__()
        self.metasurface = metasurface 
        self.renderer = renderer
        self.estimator = estimator 

        self.num = 0




    def forward(self, ref_im_list, depth_map_list, occ_im_list, normal_im_list):
        synthetic_images, illum_img = self.renderer.render(ref_im_list, depth_map_list, occ_im_list, normal_im_list)

        pred_depth = self.estimator(synthetic_images) 
        return pred_depth, synthetic_images 

    def get_meta_phase(self):
        return self.metasurface.get_phase() 

    def update_phase(self, new_phase):
        self.metasurface.update_phase(new_phase)

    def get_pattern(self):
        return self.metasurface.propagate()

    def get_estimator(self):
        return self.estimator 

