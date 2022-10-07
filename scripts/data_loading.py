import os

import argparse
import pickle
import numpy as np
from einops import rearrange
from PIL import Image


def main(cfg):
    path = cfg.path
    assert os.path.exists(path)
    assert os.path.exists(os.path.join(path, "obs.pkl"))
    assert os.path.exists(os.path.join(path, "action.pkl"))
    assert os.path.exists(os.path.join(path, "trajectory.pkl"))
    assert os.path.exists(os.path.join(path, "rgb_front"))
    assert os.path.exists(os.path.join(path, "rgb_top"))

    with open(os.path.join(path, "obs.pkl"), "rb") as f:
        obs = pickle.load(f)

    rgb_dict = {"front": [], "top": []}
    n_rgb_frames = len(os.listdir(os.path.join(path, f"rgb_front")))
    for view in ["front", "top"]:
        for idx in range(n_rgb_frames):
            # load {idx}.jpg using PIL
            rgb_dict[view].append(
                rearrange(
                    np.array(
                        Image.open(os.path.join(path, f"rgb_{view}", f"{idx}.jpg")),
                        copy=True,
                        dtype=np.uint8,
                    ),
                    "h w c -> c h w",
                )
            )
    rgb_dict = {k: np.stack(v, axis=0) for k, v in rgb_dict.items()}
    segm = obs.pop("segm")
    end_effector = obs.pop("ee")

    with open(os.path.join(path, "action.pkl"), "rb") as f:
        action = pickle.load(f)

    with open(os.path.join(path, "trajectory.pkl"), "rb") as f:
        traj_meta = pickle.load(f)

    prompt = traj_meta.pop("prompt")
    prompt_assets = traj_meta.pop("prompt_assets")

    print("Obs")
    for k, v in rgb_dict.items():
        print(f"RGB {k} view : {v.shape}")
    for k, v in segm.items():
        print(f"Segm {k} view : {v.shape}")
    print("End effector : ", end_effector.shape)
    print("-" * 50)

    print("Action")
    for k, v in action.items():
        print(f"{k} : {v.shape}")
    print("-" * 50)

    print("Prompt: ", prompt)
    print("Prompt assets keys: ", str(list(prompt_assets.keys())))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()
    main(args)
