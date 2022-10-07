import hydra
import numpy as np
from tqdm import tqdm

import vima_bench


@hydra.main(config_path=".", config_name="conf")
def main(cfg):
    kwargs = cfg.vima_bench_kwargs
    seed = kwargs["seed"]

    env = vima_bench.make(**kwargs)
    task = env.task
    oracle_fn = task.oracle(env)

    for _ in tqdm(range(999)):
        env.seed(seed)

        obs = env.reset()
        env.render()
        prompt, prompt_assets = env.get_prompt_and_assets()
        print(prompt)
        for _ in range(task.oracle_max_steps):
            oracle_action = oracle_fn.act(obs)
            # clamp action to valid range
            oracle_action = {
                k: np.clip(v, env.action_space[k].low, env.action_space[k].high)
                for k, v in oracle_action.items()
            }
            obs, reward, done, info = env.step(action=oracle_action, skip_oracle=False)
            if done:
                break

        seed += 1


if __name__ == "__main__":
    main()
