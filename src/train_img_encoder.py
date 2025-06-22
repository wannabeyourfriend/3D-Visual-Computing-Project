import os
import argparse
import torch
import torch.nn as nn
import torch.utils.tensorboard
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from utils.misc import *
from utils.data import *
from models.encoders.image_encoder import ImageAutoEncoder

parser = argparse.ArgumentParser()
parser.add_argument('--latent_dim', type=int, default=256)

parser.add_argument('--dataset_path', type=str, default='./data/airplane_img')
parser.add_argument('--img_size', type=int, default=224)
parser.add_argument('--train_batch_size', type=int, default=64)
parser.add_argument('--val_batch_size', type=int, default=32)

parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--max_grad_norm', type=float, default=10)
parser.add_argument('--epochs', type=int, default=50)

parser.add_argument('--seed', type=int, default=3047)
parser.add_argument('--logging', type=eval, default=True, choices=[True, False])
parser.add_argument('--log_root', type=str, default='./logs_img_encoder')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--val_freq', type=int, default=1)
parser.add_argument('--tag', type=str, default=None)
args = parser.parse_args()
seed_all(args.seed)

if args.logging:
    log_dir = get_new_log_dir(args.log_root, prefix='IMG_', postfix='_' + args.tag if args.tag is not None else '')
    logger = get_logger('train', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    ckpt_mgr = CheckpointManager(log_dir)
else:
    logger = get_logger('train', None)
    writer = BlackHole()
    ckpt_mgr = BlackHole()
logger.info(args)

logger.info('加载数据集...')

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(args.img_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(args.img_size),
    transforms.CenterCrop(args.img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = ImageFolder(os.path.join(args.dataset_path, 'train'), transform=train_transform)
val_dataset = ImageFolder(os.path.join(args.dataset_path, 'val'), transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False, num_workers=4)

logger.info('构建模型...')
model = ImageAutoEncoder(latent_dim=args.latent_dim).to(args.device)
logger.info(repr(model))

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

criterion = nn.MSELoss()

def train_epoch(epoch):
    model.train()
    total_loss = 0
    for batch_idx, (data, _) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch}')):
        data = data.to(args.device)
        
        optimizer.zero_grad()
        recon, _ = model(data)
        
        loss = criterion(recon, data)
        
        loss.backward()
        orig_grad_norm = clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            logger.info(f'Train Epoch: {epoch} [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.6f} Grad: {orig_grad_norm:.4f}')
    
    avg_loss = total_loss / len(train_loader)
    logger.info(f'====> Epoch: {epoch} Average loss: {avg_loss:.6f}')
    writer.add_scalar('train/loss', avg_loss, epoch)
    writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)
    
    return avg_loss

def validate(epoch):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data, _ in tqdm(val_loader, desc='Validate'):
            data = data.to(args.device)
            recon, _ = model(data)
            loss = criterion(recon, data)
            total_loss += loss.item()
            
    avg_loss = total_loss / len(val_loader)
    logger.info(f'====> Validation set loss: {avg_loss:.6f}')
    writer.add_scalar('val/loss', avg_loss, epoch)
    
    if epoch % 5 == 0:
        with torch.no_grad():
            data = next(iter(val_loader))[0][:8].to(args.device)
            recon, _ = model(data)
            comparison = torch.cat([data, recon])
            writer.add_images('reconstruction', comparison, epoch)
    
    return avg_loss

logger.info('开始训练...')
best_loss = float('inf')
for epoch in range(1, args.epochs + 1):
    train_loss = train_epoch(epoch)
    val_loss = validate(epoch)
    scheduler.step()
    
    if val_loss < best_loss:
        best_loss = val_loss
        ckpt_mgr.save(model, args, epoch, 0, val_loss)
    else:
        ckpt_mgr.save(model, args, epoch, 0, val_loss)

logger.info('训练完成!')