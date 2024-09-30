import os
from PIL import Image
from collections import defaultdict

def get_image_sizes(folder):
    size_count = defaultdict(int)

    for filename in os.listdir(folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            file_path = os.path.join(folder, filename)
            try:
                with Image.open(file_path) as img:
                    size = img.size
                    size_count[size] += 1
            except Exception as e:
                print(f"无法打开文件 {file_path}: {e}")

    return size_count

folders = ['/home/guantp/Infrared/datasets/SIRST/images/', 
           '/home/guantp/Infrared/datasets/MWIRSTD/imgs/',
           '/home/guantp/Infrared/datasets/NUDT-SIRST/images/',
           '/home/guantp/Infrared/datasets/IRSTD-1k/IRSTD1k_Img/']

for folder in folders:
    sizes = get_image_sizes(folder)
    with open('/home/guantp/Infrared/SIRST/datasets/dataset_statistics.txt', 'a+') as f:
        f.write(f'---------------------- {folder} ----------------------\n')
        for size, count in sizes.items():
            f.write(f"size: {size}, count: {count}\n")
