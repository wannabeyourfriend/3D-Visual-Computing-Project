import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ImageEncoder(nn.Module):
    def __init__(self, out_dim=256, pretrained=True):
        super().__init__()
        resnet = models.resnet50(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.projection = nn.Linear(2048, out_dim)
        
    def forward(self, x):
        features = self.backbone(x)  # (B, 512, 1, 1)
        features = features.view(features.size(0), -1)  # (B, 512)
        out = self.projection(features)  # (B, out_dim)
        return out
    
    
class ImageAutoEncoder(nn.Module):
    
    def __init__(self, latent_dim=256):
        super().__init__()
        self.encoder = ImageEncoder(out_dim=latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512 * 7 * 7),
            nn.ReLU(True),
            nn.Unflatten(1, (512, 7, 7)), # (B, 512, 7, 7)
            
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1), # -> (B, 256, 14, 14)
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # -> (B, 128, 28, 28)
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # -> (B, 64, 56, 56)
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), # -> (B, 32, 112, 112)
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1), # -> (B, 3, 224, 224)
            nn.Tanh()
        )
    
    def forward(self, x):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon, latent
    
    def encode(self, x):
        return self.encoder(x)
    

class Image_Feature(nn.Module):
    def __init__(self, in_channels=3, out_channels=96, depth=3, expansion_factor=4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stem = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)

        self.blocks = nn.Sequential(
            *[ConvNeXtBlock(dim=out_channels, expansion_factor=expansion_factor) for _ in range(depth)]
        )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)

        x = self.stem(x)
        x = self.blocks(x)

        x = x.permute(0, 2, 3, 1)
        return x

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, expansion_factor=4):
        super().__init__()
        hidden_dim = int(dim * expansion_factor)

        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # 深度卷积
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, hidden_dim)  
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(hidden_dim, dim)
        self.drop_path = nn.Identity() 

    def forward(self, x):
        input = x
        x = self.dwconv(x)

        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)

        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        x = x.permute(0, 3, 1, 2)

        x = input + self.drop_path(x) 
        return x
