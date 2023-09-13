#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: otter_flamingo
@author: ImKe at 2023/05/18
@email: tuisaac163@gmail.com
@feature: #Enter features here
"""

import argparse, sys, re, os
from typing import List

from PIL import Image
import torch
from eval_model import BaseEvalModel
sys.path.append(os.getcwd())
from llava.model.builder import load_pretrained_model
from llava.conversation import conv_templates, SeparatorStyle
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.utils import disable_torch_init

prefix_prompt = "Human: What are the key differences between renewable and non-renewable energy sources?###Assistant: Renewable energy sources are those that can be replenished naturally in a relatively short amount of time, such as solar, wind, hydro, geothermal, and biomass. Non-renewable energy sources, on the other hand, are finite and will eventually be depleted, such as coal, oil, and natural gas. Here are some key differences between renewable and non-renewable energy sources:\n1. Availability: Renewable energy sources are virtually inexhaustible, while non-renewable energy sources are finite and will eventually run out.\n2. Environmental impact: Renewable energy sources have a much lower environmental impact than non-renewable sources, which can lead to air and water pollution, greenhouse gas emissions, and other negative effects.\n3. Cost: Renewable energy sources can be more expensive to initially set up, but they typically have lower operational costs than non-renewable sources.\n4. Reliability: Renewable energy sources are often more reliable and can be used in more remote locations than non-renewable sources.\n5. Flexibility: Renewable energy sources are often more flexible and can be adapted to different situations and needs, while non-renewable sources are more rigid and inflexible.\n6. Sustainability: Renewable energy sources are more sustainable over the long term, while non-renewable sources are not, and their depletion can lead to economic and social instability.\n###"
def get_model(args):
    """Model Provider with tokenizer and processor.
    """
    # in case of using a pretrained model with only a MLP projector weights
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(model_path, args.model_base, model_name, args.device)
    return model, tokenizer, image_processor

class EvalModel(BaseEvalModel):
    """MiniGPT4 model evaluation.
    """
    def __init__(self, args):
        # load model
        super().__init__(args)
        self.device = args.device if args.device >= 0 else "cpu"
        print("Initializing LLaVA Model..")
        self.model, self.tokenizer, self.image_processor = get_model(args)
        if self.tokenizer.eos_token == "":
            self.tokenizer.eos_token = self.tokenizer.pad_token = "</s>"
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.conv_mode = args.conv_mode

        print("Finsh Initializing Image Encoder..")
        self.model.to(self.device)
        self.model.eval()



    def _prepare_images(self, batch: List[List[Image.Image]]) -> List:
        """Preprocess images and stack them.

        Args:
            batch: A list of lists of images.

        Returns:
            A Tensor of shape
            (batch_size, images_per_example, frames, channels, height, width).
        """
        images_per_example = max(len(x) for x in batch)
        assert images_per_example==1, "only support one image per conversation for now"
        batch_images = []
        for iexample, example in enumerate(batch):
            image_tensor = self.image_processor.preprocess(example[0], return_tensors='pt')['pixel_values'][0]
            images = image_tensor.unsqueeze(0).half().to(self.device)
            batch_images.append(images)
        return batch_images

    def get_outputs(
            self,
            batch_text: List[str],
            prompt_texts: List[List[str]],
            batch_images: List[List[Image.Image]],
            max_generation_length: int,
            min_generation_length: int,
            num_beams: int,
            length_penalty: float,
            do_sample: bool = False,
            top_k: int = 5,
            max_length: int = 512,
            temperature: float = 0.2,
    ) -> List[str]:
        assert len(batch_text)==len(batch_images)==1, "only support batch size = 1 for LLaVA inference.."
        batch_images = self._prepare_images(batch_images)
        ## for batch size = 1 only
        qs = batch_text[0]
        prompt_texts = prompt_texts[0]

        if getattr(self.model.config, 'mm_use_im_start_end', False):
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[self.conv_mode].copy()
        ## add prompts for few-shot evaluation
        if len(prompt_texts):
            for prompt_text in prompt_texts:
                conv.append_message(conv.roles[0], prompt_text[0])
                conv.append_message(conv.roles[1], prompt_text[1])
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        # __import__("ipdb").set_trace()

        input_ids = tokenizer_image_token(prompt, self.tokenizer,
                                          IMAGE_TOKEN_INDEX,
                                          return_tensors='pt').unsqueeze(0).to(self.device)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        with torch.no_grad():
                res = self.model.generate(
                    input_ids,
                    images=batch_images[0],
                    num_beams=num_beams,
                    length_penalty=length_penalty,
                    max_new_tokens=max_generation_length,
                    min_new_tokens=min_generation_length,
                    top_k=top_k,
                    do_sample=do_sample,
                    temperature=temperature,
                    stopping_criteria=[stopping_criteria],
                )

        res = res[:, len(input_ids[0]): ]
        sentences = self.tokenizer.batch_decode(res, skip_special_tokens=True)
        return sentences