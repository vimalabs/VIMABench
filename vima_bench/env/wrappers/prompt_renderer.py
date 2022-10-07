import re
import os
import time

import cv2
import gym
import numpy as np
from einops import rearrange

from ..base import VIMAEnvBase


class PromptRenderer(gym.Wrapper):
    def __init__(
        self,
        env: VIMAEnvBase,
        render_img_height: int = 128,
        font_scale: float = 0.8,
        font_weight: int = 1,
    ):
        super().__init__(env)
        self._render_img_height = render_img_height
        self._font_scale = font_scale
        self._font_weight = font_weight
        self._font = cv2.FONT_HERSHEY_COMPLEX

        self._display = Cv2Display(window_name="VIMA Task Prompt")
        self._prompt_img = None

    def reset(self, *args, **kwargs):
        rtn = self.env.reset(*args, **kwargs)
        self._prompt_img = self.get_multi_modal_prompt_img()
        return rtn

    def step(self, *args, **kwargs):
        return self.env.step(*args, **kwargs)

    def render(self, mode="human", **kwargs):
        self._display(self._prompt_img)

    def close(self):
        try:
            self._display.close()
        except:
            pass
        super().close()

    def get_prompt_img_from_text(
        self,
        text: str,
        left_margin: int = 0,
    ):
        lang_textsize = cv2.getTextSize(
            text, self._font, self._font_scale, self._font_weight
        )[0]

        text_width, text_height = lang_textsize[0], lang_textsize[1]
        lang_textX = left_margin
        lang_textY = (self._render_img_height + lang_textsize[1]) // 2

        image_size = self._render_img_height, text_width + left_margin, 3
        image = np.zeros(image_size, dtype=np.uint8)
        image.fill(255)
        text_img = cv2.putText(
            image,
            text,
            org=(lang_textX, lang_textY),
            fontScale=self._font_scale,
            fontFace=self._font,
            color=(0, 0, 0),
            thickness=self._font_weight,
            lineType=cv2.LINE_AA,
        )

        return text_img

    def get_multi_modal_prompt_img(self):
        # make prompts
        prompt, prompt_assets = self.env.get_prompt_and_assets()

        prompt_pieces = re.split(r"\{[^}]+\}", prompt)
        prompt_placeholders = re.findall(r"\{([^}]+)\}", prompt)
        image_patches = []
        combined_prompt = self.get_prompt_img_from_text(
            prompt_pieces[0],
            left_margin=5,
        )
        image_patches.append(combined_prompt)
        if len(prompt_assets) > 0:  # multi-modal prompt
            for idx, text_piece in enumerate(prompt_pieces[1:]):
                img_placeholder = prompt_assets[prompt_placeholders[idx]]["rgb"][
                    "front"
                ]

                img_placeholder = self._resize_img_to(
                    img_placeholder, height=self._render_img_height
                )
                image_patches.append(img_placeholder)
                img_text_piece = self.get_prompt_img_from_text(
                    text_piece,
                )
                image_patches.append(img_text_piece)

        combined_prompt = np.concatenate(image_patches, axis=1)
        return combined_prompt

    @staticmethod
    def _resize_img_to(
        img,
        height=None,
        width=None,
        preserve_apsect_ratio=True,
    ):
        # learn from https://www.tutorialkart.com/opencv/python/opencv-python-resize-image/
        assert height or width, "must supply at least either height or width"
        img = np.array(img)
        if img.shape[0] == 3 or img.shape[0] == 1:
            img = rearrange(img, "C H W -> H W C")
        original_h, original_w, _ = img.shape

        if height and width:
            img = cv2.resize(img, (width, height))
        elif height:
            if preserve_apsect_ratio:
                scaled_width = int(img.shape[1] / img.shape[0]) * height
                img = cv2.resize(
                    img, (scaled_width, height), interpolation=cv2.INTER_AREA
                )
            else:
                img = cv2.resize(
                    img, (original_w, height), interpolation=cv2.INTER_AREA
                )
        else:
            if preserve_apsect_ratio:
                scaled_height = int(img.shape[0] / img.shape[1] * width)
                img = cv2.resize(
                    img, (width, scaled_height), interpolation=cv2.INTER_AREA
                )
            else:
                img = cv2.resize(img, (width, original_h), interpolation=cv2.INTER_AREA)
        return img


class Cv2Display:
    def __init__(
        self,
        window_name="display",
        image_size=None,
        channel_order="auto",
        bgr2rgb=True,
        step_sleep=0,
        enabled=True,
    ):
        """
        Use cv2.imshow() to pop a window, requires virtual desktop GUI

        Args:
            channel_order: auto, hwc, or chw
            image_size: None to use the original image size, otherwise resize
            step_sleep: sleep for a few seconds
        """
        self._window_name = window_name
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        else:
            assert image_size is None or len(image_size) == 2
        self._image_size = image_size
        assert channel_order in ["auto", "chw", "hwc"]
        self._channel_order = channel_order
        self._bgr2rgb = bgr2rgb
        self._step_sleep = step_sleep
        self._enabled = enabled

    def _resize(self, img):
        if self._image_size is None:
            return img
        H, W = img.shape[:2]
        Ht, Wt = self._image_size  # target
        return cv2.resize(
            img,
            self._image_size,
            interpolation=cv2.INTER_AREA if Ht < H else cv2.INTER_LINEAR,
        )

    def _reorder(self, img):
        if self._channel_order == "chw":
            return np.transpose(img, (1, 2, 0))
        elif self._channel_order == "hwc":
            return img
        else:
            if img.shape[0] in [1, 3]:  # chw
                return np.transpose(img, (1, 2, 0))
            else:
                return img

    def __call__(self, img):
        if not self._enabled:
            return
        import torch

        # prevent segfault in IsaacGym
        display_var = os.environ.get("DISPLAY", None)
        if not display_var:
            os.environ["DISPLAY"] = ":0.0"

        if torch.is_tensor(img):
            img = img.detach().cpu().numpy()

        img = self._resize(self._reorder(img))
        if self._bgr2rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        time.sleep(self._step_sleep)
        cv2.imshow(self._window_name, img)
        cv2.waitKey(1)

        if display_var is not None:
            os.environ["DISPLAY"] = display_var

    def close(self):
        if not self._enabled:
            return
        cv2.destroyWindow(self._window_name)
