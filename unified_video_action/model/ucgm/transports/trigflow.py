import torch
import math

# # def s(t):                       # compressed time
# #     return (t + 0.008) / 1.008

# # class TrigFlow:
# #     # sqrt(1- alpha_bar(t))
# #     def alpha_in(self, t):
# #         return torch.sin(s(t) * math.pi / 2)
# #     # sqrt(alpha_bar(t))
# #     def gamma_in(self, t):
# #         return torch.cos(s(t) * math.pi / 2)
# #     def alpha_to(self, t):
# #         return torch.cos(s(t)* math.pi / 2.016) #* math.pi / 2.016
# #     def gamma_to(self, t):
# #         return -torch.sin(s(t) * math.pi / 2.016) #* math.pi / 2.016


# def s(t):                       # compressed time
#     return (t + 0.008)*math.pi / 2.016

# def alpha_bar_sqrt(t):
#     return torch.cos(s(t))  # sqrt(alpha_bar(t))

# class TrigFlow:
#     def alpha_in(self, t):
#         return torch.sin(s(t))
#     def gamma_in(self, t):
#         return torch.cos(s(t))
#     def alpha_to(self, t):
#         return torch.cos(s(t))
#     def gamma_to(self, t):
#         return -torch.sin(s(t))
#     def beta(self, t):
#         #alpha_bar_t_plus_1 = math.cos((t+0.009)/1.008 * math.pi / 2)**2
        
#         #alpha_bar_t = math.cos((t+0.008)/1.008 * math.pi / 2)**2
#         alpha_bar_t = alpha_bar_sqrt(t)**2
#         alpha_bar_t_minus_1 = alpha_bar_sqrt(torch.clamp(t-0.01, min=0))**2

#         return min(1-alpha_bar_t/alpha_bar_t_minus_1, 0.999)
#     def min_log(self, t):
#         #t = t.clamp_min(0.01)  # Avoid log(0)
#         return torch.log(self.beta(t)*(1-alpha_bar_sqrt(torch.clamp(t-0.01, min=0))**2) / (1-alpha_bar_sqrt(t)**2))

#     def max_log(self, t):
#         return math.log(self.beta(t))

import torch


class TrigFlow:

    def alpha_in(self, t):
        return torch.sin(t * 1.57)

    def gamma_in(self, t):
        return torch.cos(t * 1.57)

    def alpha_to(self, t):
        return torch.cos(t * 1.57)

    def gamma_to(self, t):
        return -torch.sin(t * 1.57)