import os
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from .dataset import synsetid_to_cate, cate_to_synsetid
from tqdm import tqdm

class ShapeNetImagePC(Dataset):
    
    def __init__(self, data_root, cates, split, scale_mode, img_size=224, transform=None):
        super().__init__()
        assert isinstance(cates, list), '`cates` must be a list of cate names.'
        assert split in ('train', 'val', 'test')
        assert scale_mode is None or scale_mode in ('global_unit', 'shape_unit', 'shape_bbox', 'shape_half', 'shape_34')
        
        self.data_root = data_root
        self.split = split
        self.scale_mode = scale_mode
        self.transform = transform

        if 'all' in cates:
            self.cate_synsetids = list(cate_to_synsetid.values())
        else:
            self.cate_synsetids = [cate_to_synsetid[s] for s in cates]
        
        self.img_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.stats = self.get_statistics()
        
        self.sample_paths = self._build_index()

    def get_statistics(self):
        stats_dir = os.path.join(self.data_root, 'stats')
        os.makedirs(stats_dir, exist_ok=True)
        if len(self.cate_synsetids) == len(cate_to_synsetid):
            stats_save_path = os.path.join(stats_dir, 'stats_all.pt')
        else:
            sorted_cates = sorted(self.cate_synsetids)
            stats_save_path = os.path.join(stats_dir, 'stats_' + '_'.join(sorted_cates) + '.pt')
        
        if os.path.exists(stats_save_path):
            return torch.load(stats_save_path)

        print(f"统计文件 {stats_save_path} 不存在，现在开始计算...")
        
        all_pc_paths = []
        for split in ['train', 'val', 'test']:
            index_file = os.path.join(self.data_root, f"{split}.txt")
            with open(index_file, 'r') as f:
                paths = f.read().splitlines()
                for p in paths:
                    synsetid = os.path.dirname(p)
                    if synsetid in self.cate_synsetids:
                        all_pc_paths.append(os.path.join(self.data_root, p, 'pointcloud.npy'))

        pointclouds = []
        for pc_path in tqdm(all_pc_paths, desc="加载点云以计算统计数据"):
            pc = torch.from_numpy(np.load(pc_path)).float()
            pointclouds.append(pc)
        
        all_points = torch.stack(pointclouds, dim=0)
        B, N, _ = all_points.size()
        mean = all_points.view(B*N, -1).mean(dim=0)
        std = all_points.view(-1).std(dim=0)
        stats = {'mean': mean, 'std': std}
        
        torch.save(stats, stats_save_path)
        print("统计数据计算并保存完毕。")
        return stats
    
    def _build_index(self):
        index_file_path = os.path.join(self.data_root, f"{self.split}.txt")
        with open(index_file_path, 'r') as f:
            all_paths = f.read().splitlines()
        if len(self.cate_synsetids) != len(cate_to_synsetid):
            filtered_paths = []
            for path in all_paths:
                synsetid = os.path.dirname(path) # e.g., 'airplane/model_id' -> 'airplane'
                if synsetid in self.cate_synsetids:
                    filtered_paths.append(path)
            print(f"[{self.split.upper()}] 使用 {len(filtered_paths)}/{len(all_paths)} 个样本 (根据所选类别过滤)。")
            return filtered_paths
        else:
            print(f"[{self.split.upper()}] 使用全部 {len(all_paths)} 个样本。")
            return all_paths

    def __len__(self):
        return len(self.sample_paths)
    
    def __getitem__(self, idx):
        relative_path = self.sample_paths[idx]
        synsetid, model_id = relative_path.split(os.sep)

        pc_path = os.path.join(self.data_root, relative_path, 'pointcloud.npy')
        img_path = os.path.join(self.data_root, relative_path, 'image.png')

        try:
            pc = torch.from_numpy(np.load(pc_path)).float()
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"错误：加载数据失败 {relative_path} - {e}")
            return self.__getitem__(np.random.randint(0, len(self)))

        if self.scale_mode == 'global_unit':
            shift = pc.mean(dim=0).reshape(1, 3)
            scale = self.stats['std'].reshape(1, 1)
        elif self.scale_mode == 'shape_unit':
            shift = pc.mean(dim=0).reshape(1, 3)
            scale = torch.sqrt(((pc - shift) ** 2).sum() / (pc.shape[0] * pc.shape[1])).reshape(1, 1)
        else:
            shift = torch.zeros([1, 3])
            scale = torch.ones([1, 1])
        
        pc = (pc - shift) / scale

        img_tensor = self.img_transform(img)
        
        data = {
            'pointcloud': pc,
            'image': img_tensor,
            'cate': synsetid_to_cate[synsetid],
            'id': model_id,
            'shift': shift,
            'scale': scale
        }

        if self.transform is not None:
            data = self.transform(data)
            
        return data