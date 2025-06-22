import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
try:
    from models.encoders.image_encoder import ImageAutoEncoder
except ImportError:
    exit()

def denormalize(tensor: torch.Tensor) -> np.ndarray:
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    if tensor.is_cuda:
        tensor = tensor.cpu()
        
    numpy_image = tensor.numpy().transpose(1, 2, 0)
    
    numpy_image = std * numpy_image + mean
    
    numpy_image = np.clip(numpy_image, 0, 1)
    
    return numpy_image

def test_reconstruction(args):
    model = ImageAutoEncoder(latent_dim=args.latent_dim)
    if not os.path.exists(args.weights_path):
        return

    try:
        checkpoint = torch.load(args.weights_path, map_location=torch.device(args.device))
        model.load_state_dict(checkpoint['state_dict'])
    except KeyError:
        return
    except Exception as e:
        return

    model.to(args.device)
    model.eval() 
    val_transform = transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if not os.path.exists(args.image_path):
        return
        
    input_image = Image.open(args.image_path).convert('RGB')

    input_tensor = val_transform(input_image)
    input_batch = input_tensor.unsqueeze(0).to(args.device)

    with torch.no_grad():
        reconstructed_tensor, _ = model(input_batch)

    reconstructed_tensor = reconstructed_tensor.squeeze(0).cpu()

    original_img_display = denormalize(input_tensor)
    reconstructed_img_display = denormalize(reconstructed_tensor)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(original_img_display)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(reconstructed_img_display)
    axes[1].set_title("Reconstructed Image")
    axes[1].axis('off')

    plt.suptitle('Image Reconstruction using AutoEncoder', fontsize=16)
    
    if args.output_path:
        output_dir = os.path.dirname(args.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        plt.savefig(args.output_path)

    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='图像自动编码器推理脚本')
    parser.add_argument('--latent_dim', type=int, default=256, help='模型的潜在维度，必须与训练时一致')
    parser.add_argument('--img_size', type=int, default=224, help='图像尺寸，必须与训练时一致')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='运行设备 (cuda 或 cpu)')
    parser.add_argument('--weights_path', type=str, default='./checkpoint/Img_encoder.pt', help='预训练模型权重的路径')
    parser.add_argument('--image_path', type=str, default='/cluster/home1/wzx/data/ShapeNet_tiny/02691156/d6b4ad58a49bb80cd13ef00338ba8c52/img_choy2016/000.jpg', help='要进行推理的单张图片的路径')
    parser.add_argument('--output_path', type=str, default='/cluster/home1/wzx/data/vis', help='保存结果图像的路径（可选）')
    
    args = parser.parse_args()
    
    test_reconstruction(args)