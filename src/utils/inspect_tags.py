import sys
import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def inspect_event_file(event_path):
    if not os.path.exists(event_path):
        print(f"错误: 文件不存在 -> {event_path}")
        return

    print(f"正在分析文件: {event_path}\n")
    
    accumulator = EventAccumulator(event_path, size_guidance={
        'tensors': 0,  # 加载所有张量
        'scalars': 0,
        'images': 0,
        'histograms': 0,
    })
    accumulator.Reload()

    tags = accumulator.Tags()

    for tag_type, tag_list in tags.items():
        print(f"--- 可用的 {tag_type.upper()} 标签 ---")
        if not tag_list:
            print("    (无)")
        else:
            for tag_name in tag_list:
                print(f"    - '{tag_name}'")
        print() # 换行

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("使用方法: python3 inspect_tags.py <path_to_your_event_file>")
    else:
        inspect_event_file(sys.argv[1])