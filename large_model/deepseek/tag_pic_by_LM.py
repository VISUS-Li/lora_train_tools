import os
from deepseek.large_model import LargeModel


def ds_process_directory(directory, _LM_instance):
    # 获取目录中所有的图片文件
    image_files = [f for f in os.listdir(directory) if
                   os.path.isfile(os.path.join(directory, f)) and f.endswith('.jpg')]

    # 遍历所有的图片文件
    for image_file in image_files:
        # 打开图片
        image_path = os.path.join(directory, image_file)

        # 使用模型进行预测
        result = _LM_instance.conversation(image_path, True)

        # 将结果保存为一个新的txt文件
        with open(os.path.join(directory, os.path.splitext(image_file)[0] + '.txt'), 'w', encoding='utf-8') as f:
            f.write(result)


if __name__ == "__main__":
    LM_instance = LargeModel()
    dir_path = r"D:\AI\training\10_hxl"
    ds_process_directory(dir_path, LM_instance)
