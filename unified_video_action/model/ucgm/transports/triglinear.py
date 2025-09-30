import torch
import math

class TrigLinear:

    def alpha_in(self, t):
        return torch.sin(t * math.pi/2)

    def gamma_in(self, t):
        return torch.cos(t * math.pi/2)

    def alpha_to(self, t):
        return 1

    def gamma_to(self, t):
        return -1
