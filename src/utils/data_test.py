import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms # <-- Import this
from pathlib import Path
import numpy as np

class ShapeNetPointCloudDataset(Dataset):
    def __init__(self, root_dir, synsets=None, transform=None):

        self.root_dir = Path(root_dir)
        self.transform = transform
        self.file_list = []
        self.synset_map = {}
        self.synset_ids = []

        if synsets is None:
            synsets_dirs = [d for d in self.root_dir.iterdir() if d.is_dir()]
            synsets = [d.name for d in synsets_dirs]

        for i, synset_id in enumerate(synsets):
            self.synset_map[synset_id] = i
            self.synset_ids.append(synset_id)
        
        print(f"正在加载类别: {self.synset_ids}")

        for synset_id in self.synset_ids:
            synset_path = self.root_dir / synset_id / "points"
            if synset_path.is_dir():
                for pts_file in synset_path.glob("*.pts"):
                    model_id = pts_file.stem
                    self.file_list.append({
                        "synset_id": synset_id,
                        "model_id": model_id,
                        "path": pts_file
                    })

        if not self.file_list:
            raise FileNotFoundError(f"在路径 '{self.root_dir}' 下没有找到任何 .pts 文件。请检查您的路径和文件结构。")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_info = self.file_list[idx]
        pts_path = file_info["path"]
        synset_id = file_info["synset_id"]
        
        try:
            points = np.loadtxt(pts_path, dtype=np.float32)
        except Exception as e:
            print(f"错误：无法加载文件 {pts_path}: {e}")
            points = np.zeros((1, 3), dtype=np.float32) 
        
        points = torch.from_numpy(points)
        label = self.synset_map[synset_id]
        
        sample = {
            'points': points, 
            'label': label,
            'synset_id': synset_id,
            'model_id': file_info['model_id']
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

class SamplePoints:
    """
    Uniformly samples a fixed number of points from a point cloud.
    """
    def __init__(self, num_points):
        self.num_points = num_points

    def __call__(self, sample):
        points = sample['points']
        current_points = points.shape[0]

        if current_points > self.num_points:
            indices = torch.randperm(current_points)[:self.num_points]
            sampled_points = points[indices]
        elif current_points < self.num_points:
            indices = torch.randint(0, current_points, (self.num_points,))
            sampled_points = points[indices]
        else:
            sampled_points = points
            
        sample['points'] = sampled_points
        return sample


SHAPENET_PATH = "/cluster/home1/wzx/data/shapenet"
NUM_POINTS = 2048

data_transform = transforms.Compose([
    SamplePoints(NUM_POINTS)
])

try:
    all_synsets = ['02691156', '02773838', '04379243', '02958343']
    dataset = ShapeNetPointCloudDataset(
        root_dir=SHAPENET_PATH, 
        synsets=all_synsets,
        transform=data_transform  # <-- Apply the transform here
    )

    print(f"\n数据集加载成功！共有 {len(dataset)} 个模型。")

    sample = dataset[0]
    points = sample['points']

    print(f"\n获取第一个样本:")
    print(f"  - 类别ID: {sample['synset_id']}")
    print(f"  - 新的点云尺寸: {points.shape}") # <-- This will now be [2048, 3]

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    first_batch = next(iter(dataloader))
    print("\nDataLoader 成功获取第一个批次!")
    print(f"  - 批次中点云的尺寸: {first_batch['points'].shape}") # <-- Should be [4, 2048, 3]
    print(f"  - 批次中标签的尺寸: {first_batch['label'].shape}") # <-- Should be [4]

except FileNotFoundError as e:
    print(e)
except Exception as e:
    print(f"An error occurred: {e}")