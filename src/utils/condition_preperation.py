import os
import h5py
import numpy as np
import shutil
from tqdm import tqdm

def restructure_dataset(h5_path, images_root, output_root):

    os.makedirs(output_root, exist_ok=True)

    train_paths = []
    val_paths = []
    test_paths = []

    with h5py.File(h5_path, 'r') as hf:
        categories = list(hf.keys())

        for category_id in tqdm(categories, desc="所有类别"):
            category_group = hf[category_id]
            
            for split in ['train', 'val', 'test']:
                if f'{split}_ids' not in category_group or f'{split}_points' not in category_group:
                    print(f"警告: 在类别 {category_id} 中找不到 {split} 数据，跳过。")
                    continue
                
                ids_dataset = category_group[f'{split}_ids']
                points_dataset = category_group[f'{split}_points']

                for i in tqdm(range(len(ids_dataset)), desc=f"处理 {category_id} [{split}]", leave=False):
                    model_id = ids_dataset[i].decode('utf-8')
                    point_cloud = points_dataset[i] 
                    src_image_path = os.path.join(images_root, f"{model_id}.png")
                    dest_folder = os.path.join(output_root, category_id, model_id)
                    dest_pc_path = os.path.join(dest_folder, 'pointcloud.npy')
                    dest_image_path = os.path.join(dest_folder, 'image.png')
                    os.makedirs(dest_folder, exist_ok=True)
                    np.save(dest_pc_path, point_cloud)
                    if os.path.exists(src_image_path):
                        shutil.copy(src_image_path, dest_image_path)
                    else:
                        print(f"警告: 找不到图片文件 {src_image_path}，将只保存点云。")
                    
                    relative_path = os.path.join(category_id, model_id)
                    if split == 'train':
                        train_paths.append(relative_path)
                    elif split == 'val':
                        val_paths.append(relative_path)
                    else: 
                        test_paths.append(relative_path)
    print("\n数据文件重组完成，开始写入索引文件...")

    split_map = {'train': train_paths, 'val': val_paths, 'test': test_paths}
    for split_name, paths in split_map.items():
        index_file_path = os.path.join(output_root, f"{split_name}.txt")
        with open(index_file_path, 'w') as f:
            f.write('\n'.join(paths))
        print(f"'{index_file_path}' 已创建，包含 {len(paths)} 条记录。")

    print("\n✅ 所有工作已完成！")


if __name__ == '__main__':
    H5_FILE_PATH = './data/shapenet_conditional_2048.h5'
    IMAGES_FOLDER_PATH = './data/images'
    OUTPUT_FOLDER_PATH = './processed_data'
    restructure_dataset(H5_FILE_PATH, IMAGES_FOLDER_PATH, OUTPUT_FOLDER_PATH)