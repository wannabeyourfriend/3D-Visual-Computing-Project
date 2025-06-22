# 文件名: beautiful_visualizer.py
# 描述: 一个用于点云的、极具美感且高性能的可视化脚本。

import open3d as o3d
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from typing import Optional

def create_beautiful_visualization(
    npy_file_path: str,
    point_size: float = 3.0,
    bg_color: tuple = (0.1, 0.1, 0.1, 1.0),
    cmap_name: str = 'viridis'
):
    """
    以极具美感的方式加载并可视化一个.npy点云文件。

    该函数通过以下方式提升视觉效果：
    - 移除耗时的法线计算，追求高性能。
    - 根据点的高度(Z轴)应用平滑的色彩梯度，增强深度感知。
    - 使用现代渲染器将点渲染为平滑的圆形。
    - 采用优雅的深色背景，使点云主体更突出。
    - 实现流畅的自动旋转动画。

    Args:
        npy_file_path (str): .npy 文件的路径。
        point_size (float): 渲染出的点/球的大小。
        bg_color (tuple): 背景颜色 (R, G, B, Alpha)。
        cmap_name (str): Matplotlib 的色彩映射方案 (例如: 'viridis', 'plasma', 'coolwarm')。
    """
    # 1. 加载点云
    print(f"✅ 正在加载点云: {npy_file_path}")
    if not os.path.exists(npy_file_path):
        print(f"❌ 错误: 文件未找到 -> {npy_file_path}")
        return
    
    try:
        points = np.load(npy_file_path)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("点云数据必须是一个 (N, 3) 的数组。")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
    except Exception as e:
        print(f"❌ 错误: 加载或解析文件失败。 {e}")
        return

    # 2. 创建色彩梯度 (核心美学步骤)
    print("🎨 正在应用高度色彩梯度...")
    z_coords = np.asarray(pcd.points)[:, 2]
    # 将高度值归一化到 [0, 1] 区间
    z_normalized = (z_coords - z_coords.min()) / (z_coords.max() - z_coords.min())
    
    # 从Matplotlib获取色彩映射
    cmap = plt.get_cmap(cmap_name)
    colors = cmap(z_normalized)[:, :3]  # 忽略alpha通道
    
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # 3. 设置并运行现代可视化器
    print("🚀 启动现代可视化窗口...")

    # 使用 o3d.visualization.draw 实现，它更现代且支持更好的渲染
    # 我们将通过一个回调函数来实现旋转动画
    def rotate_view(vis):
        ctr = vis.get_view_control()
        ctr.rotate(10.0, 0.0) # 调整旋转速度
        return False

    o3d.visualization.draw_geometries_with_animation_callback(
        [pcd], 
        rotate_view,
        window_name="绝美的点云可视化",
        width=1280,
        height=720,
    )

    # 或者，如果你更喜欢手动旋转，可以使用下面这行更简洁的代码
    # o3d.visualization.draw(
    #     pcd,
    #     point_size=point_size,
    #     bg_color=bg_color,
    #     show_skybox=False,
    #     window_name="绝美的点云可视化"
    # )

    print("✅ 可视化结束。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="一个用于点云的、极具美感且高性能的可视化脚本。",
        formatter_class=argparse.RawTextHelpFormatter # 保持帮助文本格式
    )
    parser.add_argument(
        '--file', 
        type=str, 
        required=True, 
        help="指向 .npy 点云文件的路径。"
    )
    parser.add_argument(
        '--size',
        type=float,
        default=3.0,
        help="渲染出的点的大小，建议范围 [1.0 - 5.0]。"
    )
    parser.add_argument(
        '--cmap',
        type=str,
        default='viridis',
        help="""选择色彩映射方案，为点云带来不同风格:
- 'viridis' (默认): 专业且色盲友好的蓝绿色系
- 'plasma': 鲜艳的紫红色系
- 'coolwarm': 冷暖对比分明
- 'jet': 经典的彩虹色系
"""
    )
    
    args = parser.parse_args()
    
    create_beautiful_visualization(
        npy_file_path=args.file,
        point_size=args.size,
        cmap_name=args.cmap
    )