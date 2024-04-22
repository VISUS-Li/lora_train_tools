from LM_Enum import LM_type
from deepseek.large_model import LargeModel
from deepseek.tag_pic_by_LM import ds_process_directory
from qwen_vl.tag_pic_by_LM import process_directory
from qwen_vl.qwen_vl_chat import LM

using_LM = LM_type.deepseek

dir_path = r"D:\AI\training\10_hxl"

if __name__ == "__main__":
    if using_LM == LM_type.qwen:
        qwen_LM_instance = LM()
        process_directory(dir_path, qwen_LM_instance)
    elif using_LM == LM_type.deepseek:
        LM_instance = LargeModel()
        ds_process_directory(dir_path, LM_instance)