from __future__ import annotations

import pybullet as p

from ..base import VIMAEnvBase


class GUIRecorder(VIMAEnvBase):
    def __init__(
        self,
        video_name: str,
        video_fps: int = 60,
        video_height: int = 480,
        video_width: int = 640,
        **kwargs,
    ):
        self._video_name = video_name
        self._video_fps = video_fps
        self._video_height = video_height
        self._video_width = video_width
        kwargs["display_debug_window"] = True
        super().__init__(**kwargs)

    def connect_pybullet_hook(self, display_debug_window: bool):
        return p.connect(
            p.GUI,
            options=f"--width={self._video_width} --height={self._video_height} --mp4={self._video_name} --mp4fps={self._video_fps}",
        )
