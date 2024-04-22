# 通过大模型筛选符合条件的图片
import fnmatch
import os
import shutil
from qwen_vl import qwen_vl_chat


def filter_image(_origin_folder_path, _dst_folder_path, LM_instance):
    # 获取目录下所有的图片路径
    images_path = find_images(_origin_folder_path)
    # 创建通过筛选的图片存储目录
    LM_filter_directory = f"{_dst_folder_path}\\LM_filter"
    os.makedirs(LM_filter_directory, exist_ok=True)
    # 创建未通过筛选的图片存储目录
    LM_no_filter_directory = f"{_dst_folder_path}\\LM_on_filter"
    os.makedirs(LM_no_filter_directory, exist_ok=True)

    for image_path in images_path:
        # 获取文件名
        file_name = os.path.basename(image_path)
        # 将图片丢给大模型筛选
        available = LM_instance.conversation(image_path)
        if available:
            # 如果该图片可用，则拷贝到目标目录
            lm_image_path = os.path.join(LM_filter_directory, file_name)
            shutil.copy(image_path, lm_image_path)
        else:
            # 如果该图片可用，则拷贝到目标目录
            lm_no_image_path = os.path.join(LM_no_filter_directory, file_name)
            shutil.copy(image_path, lm_no_image_path)


def find_images(directory):
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.tiff', '*.webp']
    images = []

    for root, dirnames, filenames in os.walk(directory):
        for extension in extensions:
            for filename in fnmatch.filter(filenames, extension):
                images.append(os.path.join(root, filename))

    return images


if __name__ == "__main__":
    origin_folder_path = r"D:\lnt\train\project\huaxiaolao\labelframes"
    dst_folder_path = r"D:\lnt\train\project\huaxiaolao\labelframes\LM"
    LM_instance = qwen_vl_chat.LM()
    filter_image(origin_folder_path, dst_folder_path, LM_instance)
