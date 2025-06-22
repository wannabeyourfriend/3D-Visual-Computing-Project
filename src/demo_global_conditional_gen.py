import argparse
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from models.conditional_autoencoder import ConditionalAutoEncoder

def generate_point_cloud(args):
    device = torch.device(args.device)
    model = ConditionalAutoEncoder(args).to(device)
    
    try:
        ckpt = torch.load(args.ckpt_path, map_location=device, weights_only=False)
        model_state_dict = ckpt['state_dict']
        model.load_state_dict(model_state_dict)

    except Exception as e:
        return

    model.eval() 
    preprocess = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
    ])

    try:
        img = Image.open(args.img_path).convert('RGB')
        img_tensor = preprocess(img).unsqueeze(0).to(device)
    except FileNotFoundError:
        return

    with torch.no_grad():
        condition = model.encode_img(img_tensor)
        z = torch.randn(1, args.latent_dim).to(device)
        generated_pc = model.decode(z, condition, args.num_points, flexibility=args.flexibility)

    pc_np = generated_pc.squeeze(0).cpu().numpy()

    output_path = args.output_path or 'generated_point_cloud.xyz'
    if not output_path.endswith('.xyz'):
        output_path += '.xyz'
        
    np.savetxt(output_path, pc_np, fmt='%.6f')
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Conditional generation")
    parser.add_argument('--img_path', type=str, required=True)
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--latent_dim', type=int, default=256,)
    parser.add_argument('--condition_dim', type=int, default=256,)
    parser.add_argument('--residual', type=eval, default=True, choices=[True, False])
    parser.add_argument('--num_steps', type=int, default=100)
    parser.add_argument('--beta_1', type=float, default=1e-4)
    parser.add_argument('--beta_T', type=float, default=0.02)
    parser.add_argument('--sched_mode', type=str, default='linear')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--num_points', type=int, default=204)
    parser.add_argument('--output_path', type=str, default='./outputs/demo_con_gen_pc.txt')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    args = parser.parse_args()
    args.freeze_img_encoder = False
    generate_point_cloud(args)