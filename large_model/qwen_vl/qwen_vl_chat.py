import re

from modelscope import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from transformers.generation import GenerationConfig
import torch

torch.manual_seed(1234)


class LM(object):
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_id = "qwen/Qwen-VL-Chat"
        self.revision = 'v1.1.0'
        self.model_dir = self.download_LM()
        self.load_LM()
        self.init_history = self.init_conversation()

    def download_LM(self):
        model_dir = snapshot_download(self.model_id, revision= self.revision)
        return model_dir

    def load_LM(self):
        # 请注意：分词器默认行为已更改为默认关闭特殊token攻击防护。
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, trust_remote_code=True)

        # 打开bf16精度，A100、H100、RTX3060、RTX3070等显卡建议启用以节省显存
        # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True, bf16=True).eval()
        # 打开fp16精度，V100、P100、T4等显卡建议启用以节省显存
        # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True, fp16=True).eval()
        # 使用CPU进行推理，需要约32GB内存
        # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cpu", trust_remote_code=True).eval()
        # 默认gpu进行推理，需要约24GB显存
        self.model = AutoModelForCausalLM.from_pretrained(self.model_dir, device_map="cuda",
                                                          trust_remote_code=True).eval()

        # 可指定不同的生成长度、top_p等相关超参（transformers 4.32.0及以上无需执行此操作）
        # model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

    # 初始化对话，赋予角色和回答内容
    def init_conversation(self):
        # 第一轮对话
        text = ("You are a master of picture recognition and can accurately identify the elements in pictures and give "
                "correct descriptions. When giving a description, you only need to use phrases or simple sentences to "
                "describe the elements, characters, actions, coloring status, whether it is pure line drawing, "
                "etc. contained in the picture. Use commas to separate each phrase or statement. For example, "
                "for this picture, your answer should have a format and content similar to the following "
                "description:Lying on a jade pillow, Coloring state, brown hair, a boy bare upper body, "
                "the background is plain white, whole body is trembling, two yellow text bubbles are on his right side,"
                "bubbles writing Chinese characters")
        query = self.tokenizer.from_list_format([
            {'image': f'example/hxl_example.jpg'},
            # Either a local path or an url
            {'text': text},
        ])
        response, history = self.model.chat(self.tokenizer, query=query, history=None)
        print(response)

        # 第二轮，继续强化对图片和标签的理解
        text = ("This is another style. This picture is partially colored. Only the hair is painted brown. "
                "The rest is line drawing, and there are two Chinese characters on the right, so for this picture, "
                "you should return The content should be similar to: partial coloring, line drawing, "
                "a boy lying down, sleeping, covered with a quilt, holding the edge of the quilt with his hands,"
                " squinting, head on the pillow, eyes squinted, mouth relaxed, the boy is in a relaxed state,"
                " with two Chinese characters written in the upper right corner")
        query = self.tokenizer.from_list_format([
            {'image': f'example/hxl_example2.jpg'},
            # Either a local path or an url
            {'text': text},
        ])
        response, history = self.model.chat(self.tokenizer, query=query, history=history)
        print(response)
        return history

    def conversation(self, image, history):
        text = "Refer to the above description format and content to describe the picture accordingly."
        query = self.tokenizer.from_list_format([
            {'image': image},
            {'text': text}
        ])
        response, _history = self.model.chat(self.tokenizer, query=query, history=history)
        # 删除<box></box>包裹的所有内容
        s = re.sub(r'<box>.*?</box>', '', response)

        # 保留<ref></ref>包裹的部分，同时删除标签<ref>和</ref>
        s = re.sub(r'<ref>(.*?)</ref>', r'\1', s)
        print(s)
        return s, _history
