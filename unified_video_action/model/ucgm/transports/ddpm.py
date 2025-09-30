import torch
import math

C = (math.pi/2)/1.008

def s(t):                       # compressed time
    return (t + 0.008) / 1.008

def alpha_bar(t):
        return torch.cos(s(t) * math.pi/2)**2

class DDPM:
    def gamma_in(self, t):
        return torch.sqrt(alpha_bar(t))

    def alpha_in(self, t):
        return torch.sqrt(1 - alpha_bar(t))

    # and the hats

    def gamma_to(self, t):
        return -torch.sin(s(t) * math.pi/2)   # = d/dt [cos(...)]
    def alpha_to(self, t):
        # derivative of sqrt(1 - cos^2) = derivative of |sin|, but on [0,1] sin>=0
        return C * torch.cos(s(t) * math.pi/2)    # = d/dt [sin(...)]

