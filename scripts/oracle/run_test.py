import hydra
import numpy as np
from tqdm import tqdm
import pybullet as p
import vima_bench
import argparse

@hydra.main(config_path=".", config_name="conf")
class RT1:
    def __init__(self):
        self.model_instance = self.load_model(model_path="xxx")

    def __call__(self, prompt, obs):
        action = self.model_instance(prompt, obs)
        return action

    def load_model(self, model_path):
        raise NotImplementedError

def main(cfg):
    kwargs = cfg.vima_bench_kwargs
    seed = kwargs["seed"]

    env = vima_bench.make(**kwargs)
    task = env.task

    rt1_model = RT1()
    rt1_model.load_model(cfg.model_path)

    # all task:
    oracle_fn = task.oracle(env)

    for eposide in tqdm(range(999)):
        env.seed(seed)
        obs = env.reset(task_name=task.task_name + "_test", pos=[0, 0, 0], rot=[0, 0, 0, 0])
        env.render()
        prompt, prompt_assets = env.get_prompt_and_assets()

        done = False

        while not done:
            last_obs = obs
            action = rt1_model(prompt, last_obs)
            obs, _, done, info = env.micro_step(action=action, skip_oracle=False,
                                                     episode=env.episode_counter,
                                                     task_name=task.task_name+"_test")

if __name__ == "__main__":
    main()
