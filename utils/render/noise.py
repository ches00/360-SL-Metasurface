import torch 

class GaussianNoise:
    def __init__(self, mean, stddev, device):
        self.mean = mean
        self.stddev = stddev
        self.device = device

    def sample(self, size):
        noise = torch.zeros(size, device=self.device).normal_(self.mean, self.stddev)
        return noise