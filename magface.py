# @title MagFace
# https://github.com/IrvingMeng/MagFace/blob/main/models/magface.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MagFaceHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.scale = 64 # like temp, for softmax
        self.easy_margin = True # True
        # https://github.com/IrvingMeng/MagFace/blob/main/run/trainer.py#L86
        self.l_a = 10 # 10 # lower bound of feature norm
        self.u_a = 110 # 110 # upper bound of feature norm
        self.l_margin = .45 # paper.4, code.45 # low bound of margin
        self.u_margin = .8 # .8 # margin slope for m'
        self.lambda_g = 20 # paper35, code20 # weight for regulariser g
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim)) # [d,n_cls]
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1)
        # nn.init.xavier_uniform_(self.weight.T)
        # emb_std = in_dim**-.5
        # nn.init.trunc_normal_(self.weight.T, mean=0, std=emb_std, a=-3*emb_std, b=3*emb_std)

    def m(self, x_norm):
        return (self.u_margin-self.l_margin)/(self.u_a-self.l_a)*(x_norm-self.l_a) + self.l_margin

    def g(self, x_norm):
        return (1/x_norm + x_norm/(self.u_a**2)).mean()

    def forward(self, i): # [b]
        return self.weight.T[i] # [b,d]

    def loss(self, x, y): # [b,d], [b]
        x_norm = torch.norm(x, dim=1, keepdim=True).clamp(self.l_a, self.u_a)
        ada_margin = self.m(x_norm)
        cos_m, sin_m = torch.cos(ada_margin), torch.sin(ada_margin)
        cos_theta = F.normalize(x) @ F.normalize(self.weight, dim=0) # [b,d]@[d,n_cls]=[n,n_cls]
        sin_theta = (1-cos_theta**2)**.5
        cos_theta_m = cos_theta * cos_m - sin_theta * sin_m
        if self.easy_margin:
            cos_theta_m = torch.where(cos_theta > 0, cos_theta_m, cos_theta)
        else:
            mm = torch.sin(torch.pi - ada_margin) * ada_margin
            threshold = torch.cos(torch.pi - ada_margin)
            cos_theta_m = torch.where(cos_theta > threshold, cos_theta_m, cos_theta - mm)
        cos_theta, cos_theta_m = self.scale*cos_theta, self.scale*cos_theta_m
        one_hot = torch.zeros_like(cos_theta).scatter(1, y.view(-1,1), 1)
        output = one_hot * cos_theta_m + (1-one_hot) * cos_theta
        return F.cross_entropy(output, y) + self.lambda_g * self.g(x_norm)

# b,d = 3,4
# num_cls = 2
# x = torch.rand(b,d)*50
# y = torch.randint(0,num_cls,(b,))

# model = MagFaceHead(d, num_cls)
# loss = model.loss(x, y)
# print(loss)
# with torch.no_grad(): emb = model([0,1]).detach()
# print(emb)

