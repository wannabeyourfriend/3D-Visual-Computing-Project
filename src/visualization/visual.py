import os
os.environ['OPEN3D_CPU_RENDERING'] = 'True'

import open3d as o3d
import numpy as np
import argparse
import time
import imageio.v2 as imageio
from tqdm import tqdm
import shutil

def create_sphere_mesh(
    npy_file_path: str,
    sphere_radius: float = 0.1,
    sphere_resolution: int = 10,
    color: tuple = (0.5, 0.2, 0.5) 
) -> o3d.geometry.TriangleMesh:
    print(f"✅ 正在加载点云文件: {npy_file_path}")
    if not os.path.exists(npy_file_path):
        print(f"❌ 错误: 文件未找到 -> {npy_file_path}")
        return None
    points = np.load(npy_file_path)
    point_count = len(points)
    print(f"   -> 点云加载成功，共 {point_count} 个点。")

    print("\n⏳ 开始实例化真实三维球体... 这可能会非常耗时，请耐心等待。")
    print(f"   - 球体半径: {sphere_radius}")
    print(f"   - 球体精度: {sphere_resolution}")

    start_time = time.time()
    
    sphere_template = o3d.geometry.TriangleMesh.create_sphere(
        radius=sphere_radius, resolution=sphere_resolution
    )
    sphere_template.compute_vertex_normals()

    list_of_spheres = []
    for point in tqdm(points, desc="  实例化球体"):
        sphere_copy = o3d.geometry.TriangleMesh(sphere_template)
        sphere_copy.translate(point)
        list_of_spheres.append(sphere_copy)
    
    print("\n✅ 所有球体实例化完成。正在合并网格...")
    final_mesh = sum(list_of_spheres, o3d.geometry.TriangleMesh())

    final_mesh.paint_uniform_color(color)
    final_mesh.compute_vertex_normals()

    end_time = time.time()
    print(f"✅ 网格处理完成，总耗时: {end_time - start_time:.2f} 秒。")
    
    return final_mesh

def generate_rotation_gif(
    mesh: o3d.geometry.TriangleMesh,
    output_path: str,
    num_frames: int,
    bg_color: tuple,
    elevation_deg: float
):
    print(f"\n⏳ 开始生成GIF动画: {output_path}")
    print(f"   - 相机俯视角度: {elevation_deg} 度")

    temp_dir = "temp_gif_frames"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=800, height=800)
    vis.add_geometry(mesh)
    
    opt = vis.get_render_option()
    opt.background_color = np.asarray(bg_color)
    opt.light_on = True

    view_control = vis.get_view_control()
    
    params = view_control.convert_to_pinhole_camera_parameters()
    
    angle_rad = -np.deg2rad(elevation_deg)
    rotation_matrix = o3d.geometry.get_rotation_matrix_from_xyz((angle_rad, 0, 0))
    current_rotation = params.extrinsic[:3, :3]
    new_rotation = rotation_matrix @ current_rotation
    
    translation = params.extrinsic[:3, 3]

    new_extrinsic = np.eye(4)
    new_extrinsic[:3, :3] = new_rotation
    new_extrinsic[:3, 3] = translation
    
    # params.extrinsic = new_extrinsic
    flip_transform = np.array([
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    flipped_extrinsic = flip_transform @ new_extrinsic
    params.extrinsic = flipped_extrinsic
    
    view_control.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)
    view_control.set_zoom(0.8)

    image_paths = []
    
    print(f"   -> 正在渲染 {num_frames} 帧...")
    for i in tqdm(range(num_frames), desc="  渲染帧"):
        view_control.rotate(5.0, 0.0)
        vis.poll_events()
        vis.update_renderer()
        
        frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
        vis.capture_screen_image(frame_path, do_render=True)
        image_paths.append(frame_path)

    vis.destroy_window()

    print(f"   -> 正在将帧合成为GIF...")
    with imageio.get_writer(output_path, mode="I", duration=50, loop=0) as writer:
        for filename in tqdm(image_paths, desc="  合成GIF"):
            image = imageio.imread(filename)
            writer.append_data(image)

    shutil.rmtree(temp_dir)
    print(f"✨ GIF成功保存到: {output_path}")

def display_interactively(mesh: o3d.geometry.TriangleMesh):
    """
    打开一个交互式窗口来显示模型。
    """
    print("\n🚀 启动交互式可视化窗口...")
    o3d.visualization.draw(
        mesh,
        window_name="物理真实感球体渲染",
        bg_color=(1.0, 1.0, 1.0, 1.0),
        show_skybox=False,
        show_ui=True,
    )
    print("✅ 可视化结束。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="为点云中的每个点渲染一个物理真实的球体，并可选择生成旋转GIF。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--file', type=str, required=True, help="指向 .npy 点云文件的路径。")
    parser.add_argument('--radius', type=float, default=0.01, help="每个物理球体的半径。")
    parser.add_argument('--resolution', type=int, default=5, help="每个球体的精细度。")
    parser.add_argument('--gif', action='store_true', help="激活此标志以生成GIF而不是打开交互式窗口。")
    parser.add_argument('--output', type=str, default='./data/demo0.gif', help="输出GIF文件的名称。")
    parser.add_argument('--frames', type=int, default=120, help="GIF动画的总帧数。")
    parser.add_argument('--elevation', type=float, default=0.0, help="GIF中相机的俯视角度（度）。")

    args = parser.parse_args()
    
    final_mesh = create_sphere_mesh(
        npy_file_path=args.file,
        sphere_radius=args.radius,
        sphere_resolution=args.resolution
    )

    if final_mesh is None or final_mesh.is_empty():
        print("❌ 无法创建模型，程序退出。")
    else:
        if args.gif:
            generate_rotation_gif(
                mesh=final_mesh,
                output_path=args.output,
                num_frames=args.frames,
                bg_color=(1.0, 1.0, 1.0),
                elevation_deg=args.elevation
            )
        else:
            display_interactively(final_mesh)