import os
import math
import argparse
import torch
import torch.utils.tensorboard
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm

from utils.image_pc_dataset import ShapeNetImagePC
from utils.misc import *
from utils.data import *
from utils.transform import *
from models.conditional_autoencoder import ConditionalAutoEncoder
from evaluation import *
from models.vae_gaussian import *
from models.vae_flow import *


parser = argparse.ArgumentParser()
parser.add_argument('--latent_dim', type=int, default=256)
parser.add_argument('--condition_dim', type=int, default=256)
parser.add_argument('--num_steps', type=int, default=100)
parser.add_argument('--beta_1', type=float, default=1e-4)
parser.add_argument('--beta_T', type=float, default=0.02)
parser.add_argument('--sched_mode', type=str, default='linear')
parser.add_argument('--flexibility', type=float, default=0.0)
parser.add_argument('--residual', type=eval, default=True, choices=[True, False])
parser.add_argument('--img_encoder_path', type=str, default='./pretrained/Img_encoder.pt', help='预训练的图像编码器路径')
parser.add_argument('--freeze_img_encoder', type=eval, default=False, choices=[True, False])
parser.add_argument('--data_root', type=str, default='./data/processed_data', help='已处理数据集的根目录')
parser.add_argument('--categories', type=str_list, default=['all'])
parser.add_argument('--scale_mode', type=str, default='shape_unit')
parser.add_argument('--img_size', type=int, default=224)
parser.add_argument('--train_batch_size', type=int, default=128)
parser.add_argument('--val_batch_size', type=int, default=32)
parser.add_argument('--num_workers', type=int, default=8, help='数据加载器的工作进程数')
parser.add_argument('--rotate', type=eval, default=False, choices=[True, False])
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--max_grad_norm', type=float, default=10)
parser.add_argument('--end_lr', type=float, default=1e-4)
parser.add_argument('--sched_start_epoch', type=int, default=150*THOUSAND)
parser.add_argument('--sched_end_epoch', type=int, default=300*THOUSAND)
parser.add_argument('--seed', type=int, default=3047)
parser.add_argument('--logging', type=eval, default=True, choices=[True, False])
parser.add_argument('--log_root', type=str, default='./logs_cond_gen')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--unconditional_steps', type=int, default=100*THOUSAND, help='无条件训练的步数')
parser.add_argument('--conditional_steps', type=int, default=200*THOUSAND, help='有条件训练的步数')
parser.add_argument('--val_freq', type=int, default=2000)
parser.add_argument('--test_freq', type=int, default=30*THOUSAND)
parser.add_argument('--ckpt_freq', type=int, default=1000, help='保存checkpoint的频率')
parser.add_argument('--test_size', type=int, default=400)
parser.add_argument('--tag', type=str, default=None)

args = parser.parse_args()
seed_all(args.seed)

# 日志和Checkpoint管理器
if args.logging:
    log_dir = get_new_log_dir(args.log_root, prefix='COND_GEN_', postfix='_' + args.tag if args.tag is not None else '')
    logger = get_logger('train', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    ckpt_mgr = CheckpointManager(log_dir)
    log_hyperparams(writer, args)
else:
    logger = get_logger('train', None)
    writer = BlackHole()
    ckpt_mgr = BlackHole()
logger.info(args)

# 数据集
logger.info('加载数据集...')
transform = None
if args.rotate:
    transform = RandomRotate(180, ['pointcloud'], axis=1)
logger.info('Transform: %s' % repr(transform))
train_dset = ShapeNetImagePC(
    data_root=args.data_root, cates=args.categories, split='train',
    scale_mode=args.scale_mode, img_size=args.img_size, transform=transform,
)
val_dset = ShapeNetImagePC(
    data_root=args.data_root, cates=args.categories, split='val',
    scale_mode=args.scale_mode, img_size=args.img_size, transform=None, 
)
train_iter = get_data_iterator(DataLoader(
    train_dset, batch_size=args.train_batch_size, num_workers=args.num_workers,
    shuffle=True, pin_memory=True,
))

# 模型和优化器
logger.info('构建模型...')
model = ConditionalAutoEncoder(args).to(args.device)
if args.img_encoder_path is not None:
    logger.info(f'加载预训练的图像编码器: {args.img_encoder_path}')
    img_encoder_ckpt = torch.load(args.img_encoder_path, map_location=args.device, weights_only=False)
    autoencoder_state_dict = img_encoder_ckpt['state_dict']
    encoder_state_dict = {
        k.replace('encoder.', '', 1): v 
        for k, v in autoencoder_state_dict.items() 
        if k.startswith('encoder.')
    }
    model.img_encoder.load_state_dict(encoder_state_dict)
    logger.info("图像编码器权重加载成功。")
    if args.freeze_img_encoder:
        logger.info('冻结图像编码器参数')
        for param in model.img_encoder.parameters():
            param.requires_grad = False
logger.info(repr(model))
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
    lr=args.lr, weight_decay=args.weight_decay
)
scheduler = get_linear_scheduler(
    optimizer,
    start_epoch=args.sched_start_epoch, end_epoch=args.sched_end_epoch,
    start_lr=args.lr, end_lr=args.end_lr
)

# --- MODIFICATION: 修改 `train` 函数以支持两阶段训练 ---
def train(it):
    batch = next(train_iter)
    x = batch['pointcloud'].to(args.device)
    img = batch['image'].to(args.device)
    loss = model.get_loss(x, img)
    optimizer.zero_grad()
    model.train()
    loss.backward()
    orig_grad_norm = clip_grad_norm_(model.parameters(), args.max_grad_norm)
    optimizer.step()
    scheduler.step()

    logger.info('[Train] Iter %04d | Loss %.6f | Grad %.4f' % (
        it, loss.item(), orig_grad_norm
    ))
    writer.add_scalar('train/loss', loss, it)
    writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
    writer.add_scalar('train/grad_norm', orig_grad_norm, it)
    writer.flush()
  
def validate_inspect(it):
    # We use shuffle=False and get the first batch to ensure the visualization is consistent across different epochs
    val_loader = DataLoader(val_dset, batch_size=4, shuffle=False)
    batch = next(iter(val_loader))
    img_batch = batch['image'].to(args.device)
    ref_pc_batch = batch['pointcloud'].to(args.device) 

    with torch.no_grad():
        model.eval()
        
        logger.info('[Inspect] Generating conditional samples for visualization...')
        
        # Always perform conditional generation
        condition = model.encode_img(img_batch)
        z = torch.randn([img_batch.size(0), args.latent_dim]).to(args.device)
        gen_pc = model.decode(z, condition, 2048, flexibility=args.flexibility)

        # Send the paired data to TensorBoard under a unified tag 'val'
        # In the TensorBoard interface, you will see these three components grouped under the same step `it`
        writer.add_images('val/input_image', img_batch, global_step=it)
        writer.add_mesh('val/generated_pc', gen_pc, global_step=it)
        writer.add_mesh('val/ground_truth_pc', ref_pc_batch, global_step=it)

    writer.flush()


def test(it):
    ref_pcs, ref_imgs = [], []
    for i, data in enumerate(val_dset):
        if i >= args.test_size: break
        ref_pcs.append(data['pointcloud'].unsqueeze(0))
        ref_imgs.append(data['image'].unsqueeze(0))
    ref_pcs = torch.cat(ref_pcs, dim=0)
    ref_imgs = torch.cat(ref_imgs, dim=0)

    gen_pcs = []
    for i in tqdm(range(0, math.ceil(args.test_size / args.val_batch_size)), 'Generate'):
        with torch.no_grad():
            start_idx, end_idx = i * args.val_batch_size, min((i + 1) * args.val_batch_size, args.test_size)
            img_batch = ref_imgs[start_idx:end_idx].to(args.device)
            condition = model.encode_img(img_batch)
            z = torch.randn([img_batch.size(0), args.latent_dim]).to(args.device)
            x = model.decode(z, condition, 2048, flexibility=args.flexibility)
            gen_pcs.append(x.detach().cpu())
    gen_pcs = torch.cat(gen_pcs, dim=0)[:args.test_size]

    with torch.no_grad():
        results = compute_all_metrics(gen_pcs.to(args.device), ref_pcs.to(args.device), args.val_batch_size)
        results = {k: v.item() for k, v in results.items()}
        jsd = jsd_between_point_cloud_sets(gen_pcs.cpu().numpy(), ref_pcs.cpu().numpy())
        results['jsd'] = jsd

    writer.add_scalar('test/Coverage_CD', results['lgan_cov-CD'], global_step=it)
    writer.add_scalar('test/MMD_CD', results['lgan_mmd-CD'], global_step=it)
    writer.add_scalar('test/1NN_CD', results['1-NN-CD-acc'], global_step=it)
    writer.add_scalar('test/JSD', results['jsd'], global_step=it)
    logger.info('[Test] Coverage | CD %.6f' % (results['lgan_cov-CD']))
    logger.info('[Test] MinMatDis | CD %.6f' % (results['lgan_mmd-CD']))
    logger.info('[Test] 1-NN-CD | %.6f' % (results['1-NN-CD-acc']))
    logger.info('[Test] JSD | %.6f' % (results['jsd']))


logger.info('开始训练...')
try:
    it = 1
    max_total_iters = args.unconditional_steps + args.conditional_steps
    
    while it <= max_total_iters:
        train(it)
        
        opt_states = {
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }
        if it % args.val_freq == 0 or it == max_total_iters:
            with torch.no_grad():
                validate_inspect(it)
        if it % args.test_freq == 0 or it == max_total_iters:
            with torch.no_grad():
                test(it)
        if it == 0 or it % args.ckpt_freq == 0 or it == max_total_iters:
            logger.info(f'[Checkpoint] 在步骤 {it} 保存模型...')
            ckpt_mgr.save(model, args, 0, others=opt_states, step=it)
        it += 1
        
        
except KeyboardInterrupt:
    logger.info('训练中断')
    logger.info(f'[Checkpoint] 在中断步骤 {it} 保存模型...')
    
    opt_states = {
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    }
    
    ckpt_mgr.save(model, args, 0, others=opt_states, step=it)