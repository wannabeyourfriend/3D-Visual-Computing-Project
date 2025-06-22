import os
import glob
import argparse
import numpy as np
import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def find_event_file(directory):
    search_path = os.path.join(directory, 'events.out.tfevents.*')
    event_files = glob.glob(search_path)
    if not event_files:
        return None
    return event_files[0]

def extract_point_clouds_from_event_file(event_path, output_dir, tag='val/pointcloud_VERTEX'):
    print(f"  -> 正在从: {os.path.basename(event_path)} 加载数据...")
    
    accumulator = EventAccumulator(event_path, size_guidance={'tensors': 0})
    accumulator.Reload()
    
    if tag not in accumulator.Tags()['tensors']:
        print(f"  -> 警告: 在文件中找不到标签(tag) '{tag}'。跳过此文件。")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    tensor_events = accumulator.Tensors(tag)
    count = 0
    for event in tensor_events:
        step = event.step
        point_clouds_batch = tf.make_ndarray(event.tensor_proto)
        
        if point_clouds_batch.ndim == 4 and point_clouds_batch.shape[0] == 1:
            point_clouds_batch = point_clouds_batch[0]
            
        for i, point_cloud in enumerate(point_clouds_batch):
            filename = f"step_{step}_sample_{i}.npy"
            output_path = os.path.join(output_dir, filename)
            np.save(output_path, point_cloud)
            count += 1
            
    print(f"  -> 提取完成！共保存了 {count} 个点云文件到 '{output_dir}'")

def main():
    parser = argparse.ArgumentParser(description="从TensorBoard event文件中全自动提取点云数据。")
    parser.add_argument('--input_dir', required=True, type=str, help="单个实验的日志目录 (例如 'logs_exp/GEN_..._airplane_200000')。")
    parser.add_argument('--output_root', required=True, type=str, help="用于存放所有提取结果的根目录。")
    parser.add_argument('--tag', type=str, default='val/pointcloud', help="TensorBoard中点云的标签。")
    
    args = parser.parse_args()

    print(f"\n正在处理目录: {os.path.basename(args.input_dir)}")

    event_file = find_event_file(args.input_dir)
    if not event_file:
        print(f"  -> 错误: 在 '{args.input_dir}' 中未找到 event 文件。")
        return

    experiment_name = os.path.basename(args.input_dir)
    specific_output_dir = os.path.join(args.output_root, experiment_name)

    extract_point_clouds_from_event_file(event_file, specific_output_dir, args.tag)

if __name__ == '__main__':
    main()