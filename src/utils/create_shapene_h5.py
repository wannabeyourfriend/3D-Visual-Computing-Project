import os
import h5py
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm


PTS_ROOT_DIR = Path("/cluster/home1/wzx/data/shapenet") 

H5_OUTPUT_PATH = Path("/cluster/home1/wzx/diffusion-point-cloud/data/shapenet_core_2048.h5")

CATEGORIES_TO_PROCESS = ['02691156', '02773838', '04379243', '02958343']

NUM_POINTS = 2048

TRAIN_VAL_TEST_SPLIT = (0.8, 0.15, 0.05)

RANDOM_SEED = 3047



def sample_points(points, num_points):
    current_points = points.shape[0]

    if current_points > num_points:
        indices = np.random.choice(current_points, num_points, replace=False)
        sampled_points = points[indices]
    elif current_points < num_points:
        indices = np.random.choice(current_points, num_points, replace=True)
        sampled_points = points[indices]
    else:
        sampled_points = points
            
    return sampled_points

def process_and_save_shapenet():
    print("开始创建 ShapeNet HDF5 数据集...")
    print(f"源目录: {PTS_ROOT_DIR}")
    print(f"输出文件: {H5_OUTPUT_PATH}")
    print(f"处理类别: {CATEGORIES_TO_PROCESS}")
    
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    with h5py.File(H5_OUTPUT_PATH, 'w') as h5_file:
        
        for synset_id in tqdm(CATEGORIES_TO_PROCESS, desc="总类别进度"):
            
            category_path = PTS_ROOT_DIR / synset_id / "points"
            if not category_path.is_dir():
                print(f"\n警告：在 {category_path} 未找到 'points' 文件夹，跳过类别 {synset_id}")
                continue

            pts_files = list(category_path.glob("*.pts"))
            if not pts_files:
                print(f"\n警告：在 {category_path} 未找到任何 .pts 文件，跳过类别 {synset_id}")
                continue
            
            random.shuffle(pts_files)
            
            num_files = len(pts_files)
            num_train = int(num_files * TRAIN_VAL_TEST_SPLIT[0])
            num_val = int(num_files * TRAIN_VAL_TEST_SPLIT[1])
            
            train_files = pts_files[:num_train]
            val_files = pts_files[num_train : num_train + num_val]
            test_files = pts_files[num_train + num_val :]
            
            splits = {
                'train': train_files,
                'val': val_files,
                'test': test_files
            }

            category_group = h5_file.create_group(synset_id)
            print(f"\n正在处理类别: {synset_id} (共 {num_files} 个文件 -> "
                  f"训练: {len(train_files)}, 验证: {len(val_files)}, 测试: {len(test_files)})")

            for split_name, file_list in splits.items():
                
                if not file_list:
                    continue

                point_clouds_for_split = []
                for pts_path in tqdm(file_list, desc=f"  - {split_name}"):
                    try:
                        raw_points = np.loadtxt(str(pts_path), dtype=np.float32)
                        processed_points = sample_points(raw_points, NUM_POINTS)
                        point_clouds_for_split.append(processed_points)
                    except Exception as e:
                        print(f"错误：无法处理文件 {pts_path}: {e}")

                if point_clouds_for_split:
                    stacked_data = np.stack(point_clouds_for_split, axis=0)
                    category_group.create_dataset(
                        split_name, 
                        data=stacked_data, 
                        compression="gzip"
                    )
    
    print("-" * 50)
    print(f"✅ HDF5 数据集创建成功！已保存至: {H5_OUTPUT_PATH}")
    print("-" * 50)

def verify_h5_file():
    if not H5_OUTPUT_PATH.exists():
        print(f"错误: H5 文件 '{H5_OUTPUT_PATH}' 不存在。")
        return
        
    print("\n正在验证生成的 HDF5 文件结构...")
    with h5py.File(H5_OUTPUT_PATH, 'r') as f:
        for synset_id in f.keys():
            print(f"/{synset_id}")
            for split_name in f[synset_id].keys():
                dataset_shape = f[synset_id][split_name].shape
                print(f"  └─ {split_name} (Shape: {dataset_shape})")


if __name__ == '__main__':
    process_and_save_shapenet()
    verify_h5_file()