import os
from PIL import Image

def clean_image_dataset(input_folder, output_folder, size=(224, 224)):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"创建输出文件夹: {output_folder}")

    supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(supported_formats):
            try:
                img_path = os.path.join(input_folder, filename)
                with Image.open(img_path) as img:
                    img = img.convert('RGB')
                    img_resized = img.resize(size, Image.Resampling.LANCZOS)
                    base, _ = os.path.splitext(filename)
                    output_filename = f"{base}.jpg"
                    output_path = os.path.join(output_folder, output_filename)
                    
                    img_resized.save(output_path, 'jpeg')
                    print(f"成功处理并保存: {output_path}")

            except Exception as e:
                print(f"处理文件 {filename} 时出错: {e}")

if __name__ == '__main__':

    source_folder = '/cluster/home1/wzx/condition-DPC/data/02691156'          # 替换为你的原始图片文件夹路径
    destination_folder = '/cluster/home1/wzx/condition-DPC/data/airplane_img' # 替换为你的输出文件夹路径
    image_size = (224, 224)                 # 目标图片大小

    clean_image_dataset(source_folder, destination_folder, image_size)
    print("\n✨ 数据集清理完成！")