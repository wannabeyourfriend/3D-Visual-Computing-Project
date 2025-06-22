import open3d as o3d
import numpy as np
import argparse
import os
import imageio

def render_to_gif(npy_file_path: str, output_path: str, sphere_size: float = 5.0, frames: int = 180, fps: int = 30):
    if not os.path.exists(npy_file_path):
        print(f"❌ 错误: 文件未找到 -> {npy_file_path}")
        return
    print(f"✅ 正在加载点云文件: {npy_file_path}")
    points = np.load(npy_file_path)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd.paint_uniform_color([0.8, 0.1, 0.1])

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Offline Render", width=800, height=600, visible=False)
    vis.add_geometry(pcd)
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.1, 0.1, 0.1])
    opt.point_size = sphere_size
    images = []
    view_control = vis.get_view_control()
    for i in range(frames):
        view_control.rotate(10.0, 0.0)
        vis.poll_events()
        vis.update_renderer()
        buffer = vis.capture_screen_float_buffer(do_render=True)
        image = np.asarray(buffer) * 255
        images.append(image.astype(np.uint8))
        print(f"  -> 已渲染帧: {i + 1}/{frames}", end='\r')

    vis.destroy_window()
    print("\n✅ 渲染完成，正在生成GIF文件...")

    imageio.mimsave(output_path, images, fps=fps)
    print(f"🎉 成功！动画已保存到: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="离线渲染点云旋转动画并保存为GIF。")
    parser.add_argument('--file', type=str, required=True, help="指向 .npy 点云文件的路径。")
    parser.add_argument('--output', type=str, required=True, help="输出的GIF文件名，例如 'rotation.gif'。")
    parser.add_argument('--size', type=float, default=5.0, help="渲染出的球体的大小。")
    args = parser.parse_args()
    
    render_to_gif(args.file, args.output, args.size)