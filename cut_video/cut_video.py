

import os
import re
from shutil import copyfile, move, rmtree
import random
import string
import subprocess
from PIL import Image
"""
pillow, opencv-python
"""


def video_to_image(video_dir, out_dir, fps=30, mode='yolo', start=0, end=30):
    """
    读取目录下的视频文件，按照指定的帧率进行采样，保存为图片
    目录结构：
    - video_dir
        - video1(*.mp4)
        - video2(*.mp)
        ...
    - out_dir
        - random_name1
            - random_name1_001.jpg
            - random_name1_002.jpg
            ...
    :param mode: 处理视频的模式，yolo生成的图片为随机字符串，slowfast图片名称为img开头同时裁剪视频时长
    :param video_dir: 存放视频的目录
    :param out_dir: 输出目录
    :param fps: 指定视频帧率
    :param start: 裁剪视频的起始时间，仅slowfast模式有效
    :param end: 裁剪视频的结束时间，仅slowfast模式有效
    """
    if mode not in ['yolo', 'slowfast']:
        raise Exception('mode must be yolo or slowfast')

    random_img_name = True if mode == 'yolo' else False
    filter_str = f"'-filter:v', 'fps={fps}, trim=start={start}:end={end}'" if mode == 'slowfast' else f"'-filter:v', 'fps={fps}'"

    video_list = os.listdir(video_dir)
    video_list = [video_dir + rf'/{i}' for i in video_list if i.split('.')[-1] in ['mov', 'avi', 'mp4', 'mpg',
                                                                                   'mpeg', 'm4v', 'wmv', 'mkv']]
    random_str = "".join(random.sample(string.ascii_letters + string.digits, 9))
    # 生成一个9位数字+字母的随机字符串作为文件夹的名字
    random_out_dir = out_dir + rf'/{random_str}'
    # 使用ffmpeg按照指定的帧率进行采样，保存为图片
    for index, video_path in enumerate(video_list):
        # outpath 后面的index始终为3位数，001 002
        out_path = f"{random_out_dir}{index+1:03}"
        img_name = random_str + f"{index + 1:02}" if random_img_name else 'img'
        # out_path = f"{out_dir}/segment{index+1}"
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        command = ['ffmpeg',
                   '-i', video_path
                   ]

        if mode == "slowfast":
            command.extend(
                ['-filter:v', f'fps={fps},trim=start={start}:end={end}'])
        else:  # if mode is "yolo"
            command.extend(
                ['-filter:v', f'fps={fps}']
            )

        command.extend(['-vcodec', 'mjpeg',
                        '-y',
                        f'{out_path}/{img_name}_%05d.jpg'])

        # command = ['ffmpeg',
        #            '-i', video_path,
        #            '-filter:v', f'fps={fps}'
        #            # '-filter:v', f'fps={fps},round=up',    # 如需截取视频时长，需使用round=up，表示向上取整
        #            # '-filter:v', 'trim=start=0:end=30',
        #            '-vcodec', 'mjpeg',
        #            '-qscale', '10',
        #            '-y',
        #            f'{out_path}/{img_name}_%05d.jpg']

        subprocess.run(command)
        print(video_path, 'done')


def extraction_image(input_dir, output_dir, step=5, change_name=True):
    """
    读取./ava/rawframes/segment* 目录下的所有图片，并且每隔 step 帧保存一张图片
    文件名可不相同，目录层级示例 ./ava/rawframes/segment1/00001.jpg ./ava/rawframes/segment2/00001.jpg
    目录结构（通常直接将函数video_to_image的输出目录作为此函数的输入目录）：
    - input_dir
        - segment1
            - img_00001.jpg
            - img_00002.jpg
            ...
        - segment2
            - img_00001.jpg
            - img_00002.jpg
            ...
    :param input_dir: 存放图片的目录
    :param output_dir: 输出目录
    :param step: 间隔帧数
    :param change_name: 是否重命名图片
    """
    if not os.path.exists(input_dir):
        print(f"{input_dir} not exists")
        return
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # 获取目录下的所有存放图片的文件夹
    dir_list = [f"{input_dir}/{i}" for i in os.listdir(input_dir)]
    # 获取每个文件夹下的所有图片
    file_list = []
    for path in dir_list:
        if os.path.isdir(path):
            img_list = [i for i in os.listdir(path) if i.split('.')[-1] in ['jpg', 'png', 'jpeg', 'PNG', 'JPG']]
            img_list.sort(key=lambda x: int(re.findall(r'\d+', x.replace('_', ''), re.S)[0]))
            file_list.append(img_list)

    # file_list.sort(key=lambda x: int(re.findall(r'\d+', x.replace('_', ''), re.S)[0]))
    print(f"每隔 {step} 帧抽取一张图片作为训练集：")
    # 遍历每个文件夹下的图片，每隔step帧保存一张图片
    for index, image_list in enumerate(file_list):
        t_index = 0
        for image in image_list[::step]:
            t_index += 1
            image_dir = f"{dir_list[index]}/{image}"
            image_dir_name = image_dir.split('/')[-2]
            # 重组图片名称 ./ava/raw/segment1/img_00001.jpg -> ./ava/labelframes/segment1_00001.jpg
            if change_name:
                print(f"{t_index} copy {image_dir} -> {output_dir}/{image_dir_name}_{image.split('_')[-1]}")
                copyfile(image_dir, f"{output_dir}/{image_dir_name}_{image.split('_')[-1]}")
            else:
                print(f"{t_index} copy {image_dir} -> {output_dir}/{image}")
                copyfile(image_dir, f"{output_dir}/{image}")
            # 提取image_dir_name中的数字，作为新的文件夹名称

            # 分文件夹存储
            # if not os.path.exists(f"{output_dir}/{image_dir_name}"):
            #     os.makedirs(f"{output_dir}/{image_dir_name}")
            # # 重组图片名称 ./ava/raw/segment1/img_00001.jpg -> ./ava/labelframes/segment1_00001.jpg
            # print(f"\rcopy {image_dir} ---> {output_dir}/{image_dir_name}/{image_dir_name}_{image.split('_')[-1]}", end='')
            # copyfile(image_dir, f"{output_dir}/{image_dir_name}/{image_dir_name}_{image.split('_')[-1]}")
    print("Done.\n")


def cut_img(input_dir: str, output_dir: str, crop_px: int = 500):
    """
    读取./ava/rawframes/segment* 目录下的所有图片，将图片裁剪为指定大小
        目录结构（通常直接将函数video_to_image的输出目录作为此函数的输入目录）：
    - input_dir
        - segment1
            - img_00001.jpg
            - img_00002.jpg
            ...
        - segment2
            - img_00001.jpg
            - img_00002.jpg
            ...
    :param input_dir:  存放图片的目录，如 ./ava/rawframes/segment1/xxx.jpg 则传入./ava/rawframes
    :param output_dir: 输出目录，将保留原始图片的目录结构
    :param crop_px: 裁剪的像素数，左右两边各crop_px个像素
    """
    if not os.path.exists(input_dir):
        print(f"{input_dir} not exists")
        return
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # 获取目录下的所有存放图片的文件夹
    dir_list = [f"{input_dir}/{i}" for i in os.listdir(input_dir) if i.split('.')[-1]]
    # 获取每个文件夹下的所有图片
    file_list = [os.listdir(i) for i in dir_list]
    # 遍历每个文件夹下的图片，每隔step帧保存一张图片
    for index, image_list in enumerate(file_list):
        for image in image_list:
            if image.split('.')[-1] not in ['jpg', 'png', 'jpeg']:
                continue
            image_dir = f"{dir_list[index]}/{image}"
            image_dir_name = image_dir.split('/')[-2]
            # 重组图片名称 ./ava/raw/segment1/img_00001.jpg -> ./ava/labelframes/segment1_00001.jpg
            print(f"\rCrop and cover {image_dir} -> {output_dir}/{image_dir_name}_{image.split('_')[-1]}")
            # 将图片拷贝至./ava/labelframes 目录下
            if not os.path.exists(f"{output_dir}/{image_dir_name}"):
                os.makedirs(f"{output_dir}/{image_dir_name}")
            img = Image.open(image_dir)
            width, height = img.size
            img = img.crop((crop_px, 0, width - crop_px, height))
            img.save(f"{output_dir}/{image_dir_name}/{image}")
    print("Done.\n")


def mask_img(input_dir: str, output_dir: str, crop_px: int = 500):
    """
    将图片两边涂黑crop_px个像素
    目录结构（通常直接将函数video_to_image的输出目录作为此函数的输入目录）：
    - input_dir
        - segment1
            - img_00001.jpg
            - img_00002.jpg
            ...
        - segment2
            - img_00001.jpg
            - img_00002.jpg
            ...

    input_dir: 存放图片的目录，如 ./ava/rawframes/segment1/xxx.jpg 则传入./ava/rawframes
    output_dir: 输出目录，将保留原始图片的目录结构
    crop_px: 涂黑的像素数，左右两边各crop_px个像素
    """
    if not os.path.exists(input_dir):
        print(f"{input_dir} not exists")
        return
    # 获取目录下的所有存放图片的文件夹
    dir_list = [f"{input_dir}/{i}" for i in os.listdir(input_dir) if i.split('.')[-1]]
    # 获取每个文件夹下的所有图片
    file_list = [os.listdir(i) for i in dir_list]
    # 遍历每个文件夹下的图片，裁剪并涂黑两边500像素后保存
    for index, image_list in enumerate(file_list):
        img_root_path = f"{dir_list[index]}/"
        output_path = f"{output_dir}/{dir_list[index].split('/')[-1]}/"
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        for image in image_list:
            if image.split('.')[-1] not in ['jpg', 'png', 'jpeg']:
                continue
            with Image.open(img_root_path+image) as img:
                width, height = img.size
                img = img.crop((crop_px, 0, width - crop_px, height))
                # 计算要加的黑色区域宽度
                width, height = img.size
                new_width = width + 2 * crop_px
                # 创建一个新的黑色背景图像
                black_img = Image.new('RGB', (new_width, height), 'black')
                # 将原始图像复制到黑色图像的中心
                black_img.paste(img, (crop_px, 0))
                black_img.save(output_path+image)
                print(f"\rProcessed image {img_root_path+image}, saved to {output_path+image}", end='')


def combine_img(input_dir, output_dir):
    """
    合并输入目录下的所有图片，输出到输出目录
    将./ava/rawframes/segment* 目录下的所有图片合并为一个文件夹，前缀为segment，index从1开始
    :param input_dir: 存放图片的目录
    :param output_dir: 输出目录
    """
    if not os.path.exists(input_dir):
        print(f"{input_dir} not exists")
        return
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # 获取目录下的所有存放图片的文件夹
    dir_list = [f"{input_dir}/{i}" for i in os.listdir(input_dir)]
    # 获取每个文件夹下的所有图片
    file_list = [os.listdir(i) for i in dir_list]
    # 遍历每个文件夹下的图片，每隔step帧保存一张图片
    new_image_index = 1
    print(f"开始合并所有训练集图片至{output_dir}")
    for index, image_list in enumerate(file_list):
        for image in image_list:
            image_dir = f"{dir_list[index]}/{image}"
            print(f"\rmove {image_dir} -> {output_dir}/img_{new_image_index:05d}.jpg", end='')
            # copyfile(image_dir, f"{output_dir}/{image}")
            move(image_dir, f"{output_dir}/img_{new_image_index:05d}.jpg")
            new_image_index += 1
    rmtree(input_dir)
    print(f"\n合并完成，删除临时文件夹{input_dir}")





if __name__ == '__main__':
    folder_name = r'D:\AI\tools\cut_video_pic\video'
    # 读取视频文件，保存为图片，SlowFast数据集中，fps需要设定为30
    video_to_image(video_dir=folder_name, out_dir=f'{folder_name}/rawframes', fps=30, mode='yolo', start=0, end=15)


    # 左右两边各裁600px，直接覆盖原有图片
    # cut_img(input_dir=f'{folder_name}/rawframes', output_dir=f'{folder_name}/rawframes', crop_px=600)

    # mask图片两边600像素
    # mask_img(input_dir=f'{folder_name}/rawframes', output_dir=f'{folder_name}/rawframes', crop_px=600)

    # 制作标注图片，每隔30帧保存一张图片
    extraction_image(input_dir=f'{folder_name}/rawframes', output_dir=f'{folder_name}/labelframes', step=50)



    # -------------------------------------------Debug------------------------------------------- #
    # video_to_image(video_dir="./ava/videos", out_dir='./ava/rawframes/tmp', fps=30)  # 读取视频文件，保存为图片，每秒30帧
    # combine_img(input_dir='./ava/rawframes/tmp', output_dir='./ava/rawframes/segment')  # 合并ffmpeg生成的图片，制作训练集
    # extraction_image(input_dir='./ava/rawframes', output_dir='./ava/labelframes', step=30)  # 制作标注图片
