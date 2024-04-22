# 大模型使用模块
import glob
import os

from modelscope import snapshot_download
from Logger import Logger

import torch
from transformers import AutoModelForCausalLM

from deepseek.ds_sdk.deepseek_vl.models import MultiModalityCausalLM, VLChatProcessor
from deepseek.ds_sdk.deepseek_vl.utils.io import load_pil_images

log = Logger('../all.log', level='info').logger
err_log = Logger('../error.log', level='error').logger


class LargeModel(object):
    def __init__(self):
        # 自动下载模型
        self.tokenizer = None
        self.vl_chat_processor = None
        self.vl_gpt = None
        model_id = "deepseek-ai/deepseek-vl-7b-chat"
        # 获取当前目录
        cache_dir = os.path.join(os.getcwd(), "deepseek")
        _model_path = os.path.join(cache_dir, model_id)
        self.model_path = _model_path
        files = glob.glob(os.path.join(_model_path, '*.safetensors'))
        if files is None or len(files) == 0:
            log.info("下载模型")
            snapshot_download(self.model_path, cache_dir=cache_dir, revision="master")
            log.info("模型下载完成")

        log.info("加载模型")
        self.load_deep_seek()
        log.info("加载模型完成")

    def load_deep_seek(self):
        # specify the path to the model
        self.vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(self.model_path)
        self.tokenizer = self.vl_chat_processor.tokenizer

        self.vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self.vl_gpt = self.vl_gpt.to(torch.bfloat16).cuda().eval()

    def conversation(self, image, bool_res=False):
        text1 = (
            "You are a master of picture recognition and can accurately identify the elements in pictures and give "
            "correct descriptions. When giving a description, you only need to use phrases or simple sentences to "
            "describe the elements, characters, actions, coloring status, whether it is pure line drawing, "
            "etc. If there are suspected Chinese characters, there is no need to explain the meaning of the "
            "characters, just label them as Chinese characters."
            " contained in the picture. Use commas to separate each phrase or statement. For example, "
            "for this picture, your answer should have a format and content similar to the following "
            "description:Lying on a jade pillow, Coloring state, brown hair, a boy bare upper body, "
            "the background is plain white, whole body is trembling, two yellow text bubbles are on his right side,"
            "bubbles writing Chinese characters")
        text2 = ("This is another style. This picture is partially colored. Only the hair is painted brown. "
                 "The rest is line drawing, and there are two Chinese characters on the right, so for this picture, "
                 "you should return The content should be similar to: partial coloring, line drawing, "
                 "a boy lying down, sleeping, covered with a quilt, holding the edge of the quilt with his hands,"
                 " squinting, head on the pillow, eyes squinted, mouth relaxed, the boy is in a relaxed state,"
                 " with two Chinese characters written in the upper right corner")
        dst_text = "Refer to the above description format and content to describe the picture accordingly"
        # 获取当前目录
        current_dir = os.getcwd()

        # 拼接路径
        example_image1 = os.path.join(current_dir, 'example', 'hxl_example.jpg')
        example_image2 = os.path.join(current_dir, 'example', 'hxl_example2.jpg')
        conversation = [
            {
                "role": "User",
                "content": f"<image_placeholder>{text1}.<image_placeholder>{text2}.\n{dst_text}:<image_placeholder>",
                "images": [example_image1, example_image2, image],
            },
            {
                "role": "Assistant",
                "content": ""
            }
        ]

        # load images and prepare for inputs
        pil_images = load_pil_images(conversation)
        prepare_inputs = self.vl_chat_processor(
            conversations=conversation, images=pil_images, force_batchify=True
        ).to(self.vl_gpt.device)

        # run image encoder to get the image embeddings
        inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**prepare_inputs)

        # run the model to get the response
        outputs = self.vl_gpt.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=1024,
            do_sample=False,
            use_cache=True,
        )

        answer = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        print(f"图片:{image}，大模型输出结果:{answer}")
        if bool_res is False:
            if "否" in answer or "no" in answer or "No" in answer or "NO" in answer:
                return False
            return True
        return answer

# ds_sdk = LargeModel()
# ds_sdk.conversation("以下问题只需要你回答是或者否，满足以下条件则回复是，不满足则回复否。这张图片是否满足以下条件：1.图片中只有一个女性人物，而没有其他人物。2. 该女性人物穿着中国古代的衣服。3. 该女性人物必须占据整张图的二分之一以上。", "./2.jpg")
