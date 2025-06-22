import os
import random
import shutil

def split_dataset(base_dir, train_ratio=0.9):
    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'val')

    if os.path.exists(train_dir) or os.path.exists(val_dir):
        print(f"'{train_dir}' 或 '{val_dir}' 已存在。")
        user_input = input("是否要删除现有文件夹并重新创建？(y/n): ").lower()
        if user_input == 'y':
            if os.path.exists(train_dir):
                shutil.rmtree(train_dir)
            if os.path.exists(val_dir):
                shutil.rmtree(val_dir)
        else:
            print("操作已取消。")
            return

    os.makedirs(train_dir)
    os.makedirs(val_dir)
    print(f"已创建文件夹: {train_dir}, {val_dir}")

    supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')
    files = [f for f in os.listdir(base_dir) if os.path.isfile(os.path.join(base_dir, f)) and f.lower().endswith(supported_formats)]
    
    if not files:
        print(f"在 '{base_dir}' 文件夹中没有找到支持的图像文件。")
        return

    random.shuffle(files)

    split_point = int(len(files) * train_ratio)

    train_files = files[:split_point]
    val_files = files[split_point:]

    print("\n开始移动训练集文件...")
    for file in train_files:
        src_path = os.path.join(base_dir, file)
        dst_path = os.path.join(train_dir, file)
        shutil.move(src_path, dst_path)
    print(f"成功移动 {len(train_files)} 个文件到 'train' 文件夹。")

    print("\n开始移动验证集文件...")
    for file in val_files:
        src_path = os.path.join(base_dir, file)
        dst_path = os.path.join(val_dir, file)
        shutil.move(src_path, dst_path)
    print(f"成功移动 {len(val_files)} 个文件到 'val' 文件夹。")


if __name__ == '__main__':

    source_directory = '/cluster/home1/wzx/condition-DPC/data/dataset-img-ae/train/images' # 替换为图片数据文件夹路径
    split_ratio = 0.9            # 9:1 的分割比例

    split_dataset(source_directory, split_ratio)
    print("\n✨ 数据集划分完成！")