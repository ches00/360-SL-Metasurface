import torch 
import numpy as np 




class Metasurface:
	def __init__(self, opt, device):
		self.device = device 
		self.N_theta = opt.N_theta 
		self.N_alpha = opt.N_alpha
		self.N_phase = opt.N_phase

		self.phase = torch.rand(self.N_phase, self.N_phase).to(device) * torch.pi*2 - torch.pi # initialization


		self.wl = opt.wave_length
		self.p = opt.pixel_pitch



	def propagate(self):		
		return torch.abs(torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(torch.exp(1j * self.phase))))) ** 2 

	def get_phase(self):
		return self.phase 

	def update_phase(self, new_phase):
		self.phase = new_phase

	
