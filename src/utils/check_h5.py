import h5py
import numpy as np

H5_PATH = '/cluster/home1/wzx/condition-DPC/data/shapenet_conditional_2048.h5'

def print_hdf5_item(name, obj):
    depth = name.count('/')
    indent = '    ' * depth  

    if isinstance(obj, h5py.Group):
        print(f"{indent}📂 {obj.name}/  (组 Group)")
    elif isinstance(obj, h5py.Dataset):
        # 如果是“数据集”（类似于文件）
        # shape 告诉我们数据的维度，dtype 告诉我们数据类型
        print(f"{indent}📄 {obj.name}  (数据集 Dataset)")
        print(f"{indent}   - 维度 (Shape): {obj.shape}")
        print(f"{indent}   - 数据类型 (dtype): {obj.dtype}")
        
        # 特别地，如果数据集很小，我们可以打印一两个样本看看
        if obj.ndim == 1 and obj.shape[0] < 10: # 如果是一维且数量小于10
             print(f"{indent}   - 内容预览: {obj[:]}")


def inspect_h5_file(file_path):
    """
    主函数，用于打开 HDF5 文件并遍历其结构。
    """
    try:
        with h5py.File(file_path, 'r') as f:
            print(f"--- 开始检查文件: {file_path} ---")
            f.visititems(print_hdf5_item)
            print(f"--- 文件检查完毕 ---")
    except Exception as e:
        print(f"打开或读取文件时出错: {e}")

# --- 运行检查 ---
if __name__ == "__main__":
    inspect_h5_file(H5_PATH)