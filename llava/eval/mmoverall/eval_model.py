import abc
import argparse
from typing import List

from PIL import Image


class BaseEvalModel(abc.ABC):
    """Base class encapsulating functionality needed to evaluate a model."""

    def __init__(self, args):
        """Initialize model.

        Args:
            args: arguments to model. These should be parsed, or if the model
                has no applicable arguments, an error should be thrown if `args`
                is non-empty.
        """

    def get_outputs(
        self,
        batch_text: List[str],
        batch_images: List[List[Image.Image]],
        max_generation_length: int,
        min_generation_length: int,
        num_beams: int,
        length_penalty: float,
        do_sample: bool,
        top_k: int,
        max_length: int,
    ) -> List[str]:
        """Get outputs for a batch of images and text.

        Args:
            batch_text: list of text strings, with the text "<image>" in place
                of any images to be included.
            batch_images: images to provide to model. Should be a list of lists,
              where each list contains the images for a single example.
            max_generation_length: maximum length of the generated caption.
                Defaults to 10.
            num_beams: number of beams to use for beam search. Defaults to 3.
            length_penalty: length penalty for beam search. Defaults to -2.0.

        Returns:
            List of decoded output strings.
        """
