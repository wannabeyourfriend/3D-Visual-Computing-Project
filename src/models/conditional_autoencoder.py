import torch
from torch.nn import Module

from .encoders import *
from .encoders.image_encoder import ImageEncoder
from .conditional_diffusion import *


class ConditionalAutoEncoder(Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        
        self.pc_encoder = PointNetEncoder(zdim=args.latent_dim)
        
        self.img_encoder = ImageEncoder(out_dim=args.condition_dim, pretrained=True)
        
        self.diffusion = ConditionalDiffusionPoint(
            net = ConditionalPointwiseNet(
                point_dim=3, 
                context_dim=args.latent_dim, 
                condition_dim=args.condition_dim,
                residual=args.residual
            ),
            var_sched = VarianceSchedule(
                num_steps=args.num_steps,
                beta_1=args.beta_1,
                beta_T=args.beta_T,
                mode=args.sched_mode
            )
        )

    def encode_pc(self, x):
        code, _ = self.pc_encoder(x)
        return code
    
    def encode_img(self, img):
        return self.img_encoder(img)

    def decode(self, code, condition, num_points, flexibility=0.0, ret_traj=False):

        return self.diffusion.sample(num_points, code, condition, flexibility=flexibility, ret_traj=ret_traj)

    def get_loss(self, x, img):

        code = self.encode_pc(x)
        condition = self.encode_img(img)
        loss = self.diffusion.get_loss(x, code, condition)
        return loss