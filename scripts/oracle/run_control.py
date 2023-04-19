import hydra
import numpy as np
from tqdm import tqdm
import pybullet as p
import vima_bench


@hydra.main(config_path=".", config_name="conf")
def main(cfg):
    kwargs = cfg.vima_bench_kwargs
    seed = kwargs["seed"]

    env = vima_bench.make(**kwargs)
    task = env.task

    # all task:

    print("task_name: ", task.task_name)
    oracle_fn = task.oracle(env)

    for eposide in tqdm(range(999)):
        env.seed(seed)

        obs = env.reset(task_name=task.task_name)
        env.render()
        prompt, prompt_assets = env.get_prompt_and_assets()

        for _ in range(task.oracle_max_steps):
            oracle_action = oracle_fn.act(obs)
            # print("obs: ", obs)
            print("oracle_action:\n")
            print(oracle_action['pose0_position'])
            print(oracle_action['pose0_rotation'])
            print(oracle_action['pose1_position'])
            print(oracle_action['pose1_rotation'])

            oracle_action = {
                k: np.clip(v, env.action_space[k].low, env.action_space[k].high) for k, v in oracle_action.items()
            }
            info = p.getLinkState(3, 0)
            pre_control_actions = [[0.5009089590173395, -0.021002068396574095, 0.41974299756782535], [0.707087814337918, -0.7069156832982121, -0.008709917171032668, -0.014872016161967223],[0.5009089590173395, -0.021002068396574095, 0.41974299756782535], [0.707087814337918, -0.7069156832982121, -0.008709917171032668, -0.014872016161967223],[0.5009089590173395, -0.021002068396574095, 0.41974299756782535], [0.707087814337918, -0.7069156832982121, -0.008709917171032668, -0.014872016161967223],[0.47563592931961296, -0.1623286587409193, 0.4227175241545642], [-0.7070215988246892, 0.7071569689949396, 0.0018118895692541969, 0.006796841432537891],[0.47563592931961296, -0.1623286587409193, 0.4227175241545642], [-0.7070215988246892, 0.7071569689949396, 0.0018118895692541969, 0.006796841432537891],[0.47563592931961296, -0.1623286587409193, 0.4227175241545642], [-0.7070215988246892, 0.7071569689949396, 0.0018118895692541969, 0.006796841432537891],[0.43263981372640575, -0.2579850697135678, 0.4099439841361778], [-0.7070030826718844, 0.7071986314046744, 7.570898919120818e-05, 0.004090366726491153],[0.43263981372640575, -0.2579850697135678, 0.4099439841361778], [-0.7070030826718844, 0.7071986314046744, 7.570898919120818e-05, 0.004090366726491153],[0.43263981372640575, -0.2579850697135678, 0.4099439841361778], [-0.7070030826718844, 0.7071986314046744, 7.570898919120818e-05, 0.004090366726491153],[0.432265829105921, -0.258711760783349, 0.2938331774166932], [-0.7070103500224592, 0.7072031834791394, 0.00011248740370519313, -9.790135178118369e-05],[0.432265829105921, -0.258711760783349, 0.2938331774166932], [-0.7070103500224592, 0.7072031834791394, 0.00011248740370519313, -9.790135178118369e-05],[0.432265829105921, -0.258711760783349, 0.2938331774166932], [-0.7070103500224592, 0.7072031834791394, 0.00011248740370519313, -9.790135178118369e-05],[0.4324555909394804, -0.2588315903515503, 0.17411472532856895], [-0.7070103430970139, 0.7072031652619044, 0.00021194716401288366, -0.00011347152034280979],[0.4324555909394804, -0.2588315903515503, 0.17411472532856895], [-0.7070103430970139, 0.7072031652619044, 0.00021194716401288366, -0.00011347152034280979],[0.4324555909394804, -0.2588315903515503, 0.17411472532856895], [-0.7070103430970139, 0.7072031652619044, 0.00021194716401288366, -0.00011347152034280979],[0.4482137829298416, -0.2732290534782973, 0.21091665347137067], [-0.7069893437544359, 0.707221614281854, 0.0005078778294051771, 0.0018434126914335031],[0.4482137829298416, -0.2732290534782973, 0.21091665347137067], [-0.7069893437544359, 0.707221614281854, 0.0005078778294051771, 0.0018434126914335031],[0.4482137829298416, -0.2732290534782973, 0.21091665347137067], [-0.7069893437544359, 0.707221614281854, 0.0005078778294051771, 0.0018434126914335031],[0.44881135986240817, -0.2746774308049141, 0.3337901836798805], [-0.7069903091241532, 0.7072222627872196, 2.8178234708752112e-05, -0.0011717630504317477],[0.44881135986240817, -0.2746774308049141, 0.3337901836798805], [-0.7069903091241532, 0.7072222627872196, 2.8178234708752112e-05, -0.0011717630504317477],[0.44881135986240817, -0.2746774308049141, 0.3337901836798805], [-0.7069903091241532, 0.7072222627872196, 2.8178234708752112e-05, -0.0011717630504317477],[0.46480604419804683, -0.22909827966381646, 0.42507500260802705], [0.7389880974813369, -0.6724244225585663, 0.0028568614573713, 0.04163923714846781],[0.46480604419804683, -0.22909827966381646, 0.42507500260802705], [0.7389880974813369, -0.6724244225585663, 0.0028568614573713, 0.04163923714846781],[0.46480604419804683, -0.22909827966381646, 0.42507500260802705], [0.7389880974813369, -0.6724244225585663, 0.0028568614573713, 0.04163923714846781],[0.5654765952831474, -0.07269242440954699, 0.428310621356436], [0.8323503061199624, -0.553473554730081, 0.008089834828734257, 0.028187704569489096],[0.5654765952831474, -0.07269242440954699, 0.428310621356436], [0.8323503061199624, -0.553473554730081, 0.008089834828734257, 0.028187704569489096],[0.5654765952831474, -0.07269242440954699, 0.428310621356436], [0.8323503061199624, -0.553473554730081, 0.008089834828734257, 0.028187704569489096],[0.6076516458538747, 0.12709223284949006, 0.4250012844931253], [0.9063454911413193, -0.42246045880468697, 0.003910439664156443, 0.007051233648593142],[0.6076516458538747, 0.12709223284949006, 0.4250012844931253], [0.9063454911413193, -0.42246045880468697, 0.003910439664156443, 0.007051233648593142],[0.6076516458538747, 0.12709223284949006, 0.4250012844931253], [0.9063454911413193, -0.42246045880468697, 0.003910439664156443, 0.007051233648593142],[0.607253323038377, 0.17286111939973686, 0.3815136376679544], [0.9191030537511439, -0.3940162700912953, -0.0003747625578306913, -0.0007842459376078606],[0.607253323038377, 0.17286111939973686, 0.3815136376679544], [0.9191030537511439, -0.3940162700912953, -0.0003747625578306913, -0.0007842459376078606],[0.607253323038377, 0.17286111939973686, 0.3815136376679544], [0.9191030537511439, -0.3940162700912953, -0.0003747625578306913, -0.0007842459376078606],[0.607684924736981, 0.17279763435515705, 0.2558840462339237], [0.9191468386266584, -0.39391506977989277, -3.715805917683934e-05, 7.390670249926208e-05],[0.607684924736981, 0.17279763435515705, 0.2558840462339237], [0.9191468386266584, -0.39391506977989277, -3.715805917683934e-05, 7.390670249926208e-05],[0.607684924736981, 0.17279763435515705, 0.2558840462339237], [0.9191468386266584, -0.39391506977989277, -3.715805917683934e-05, 7.390670249926208e-05],[0.6141871921233992, 0.09909166257192689, 0.21647671166078825], [0.8944972199772878, -0.44589357643171806, -0.014132002056328625, -0.029222054474575196],[0.6141871921233992, 0.09909166257192689, 0.21647671166078825], [0.8944972199772878, -0.44589357643171806, -0.014132002056328625, -0.029222054474575196],[0.6141871921233992, 0.09909166257192689, 0.21647671166078825], [0.8944972199772878, -0.44589357643171806, -0.014132002056328625, -0.029222054474575196],[0.5778304366974453, -0.08143687827171751, 0.30877808002998786], [0.8264962999641083, -0.5627609612342237, -0.004037338965199339, -0.013699144150323654],[0.5778304366974453, -0.08143687827171751, 0.30877808002998786], [0.8264962999641083, -0.5627609612342237, -0.004037338965199339, -0.013699144150323654],[0.5778304366974453, -0.08143687827171751, 0.30877808002998786], [0.8264962999641083, -0.5627609612342237, -0.004037338965199339, -0.013699144150323654],[0.48008988186598106, -0.22118099912075934, 0.3944599858157116], [0.7429420218629544, -0.6693557605123946, -9.933020797153274e-05, -9.029157784537373e-05],[0.48008988186598106, -0.22118099912075934, 0.3944599858157116], [0.7429420218629544, -0.6693557605123946, -9.933020797153274e-05, -9.029157784537373e-05],[0.48008988186598106, -0.22118099912075934, 0.3944599858157116], [0.7429420218629544, -0.6693557605123946, -9.933020797153274e-05, -9.029157784537373e-05]]
            control_actions_ = []

            for ind,val in enumerate(pre_control_actions):
                if ind%2==0:
                    control_actions_.append(val)
                if ind % 2 == 1:
                    control_actions_[-1] += val

            control_actions = [val for ind, val in enumerate(control_actions_) if ind%3==0]
            # print(control_actions)
            while not control_actions:
                # obs, reward, done, info = env.step(action=oracle_action, skip_oracle=False, episode=env.episode_counter, task_name=task.task_name)
                obs, reward, done, info = env.micro_step(action=control_actions[-1], skip_oracle=False, episode=env.episode_counter,
                                                   task_name=task.task_name)
                control_actions.pop()
                if done:
                    break
                import time
                time.sleep(1)
        print("episode_counter: ", env.episode_counter)
        seed += 1


if __name__ == "__main__":

    main()


'''
[0.5009089590173395, -0.021002068396574095, 0.41974299756782535], [0.707087814337918, -0.7069156832982121, -0.008709917171032668, -0.014872016161967223],[0.5009089590173395, -0.021002068396574095, 0.41974299756782535], [0.707087814337918, -0.7069156832982121, -0.008709917171032668, -0.014872016161967223],[0.5009089590173395, -0.021002068396574095, 0.41974299756782535], [0.707087814337918, -0.7069156832982121, -0.008709917171032668, -0.014872016161967223],[0.47563592931961296, -0.1623286587409193, 0.4227175241545642], [-0.7070215988246892, 0.7071569689949396, 0.0018118895692541969, 0.006796841432537891],[0.47563592931961296, -0.1623286587409193, 0.4227175241545642], [-0.7070215988246892, 0.7071569689949396, 0.0018118895692541969, 0.006796841432537891],[0.47563592931961296, -0.1623286587409193, 0.4227175241545642], [-0.7070215988246892, 0.7071569689949396, 0.0018118895692541969, 0.006796841432537891],[0.43263981372640575, -0.2579850697135678, 0.4099439841361778], [-0.7070030826718844, 0.7071986314046744, 7.570898919120818e-05, 0.004090366726491153],[0.43263981372640575, -0.2579850697135678, 0.4099439841361778], [-0.7070030826718844, 0.7071986314046744, 7.570898919120818e-05, 0.004090366726491153],[0.43263981372640575, -0.2579850697135678, 0.4099439841361778], [-0.7070030826718844, 0.7071986314046744, 7.570898919120818e-05, 0.004090366726491153],[0.432265829105921, -0.258711760783349, 0.2938331774166932], [-0.7070103500224592, 0.7072031834791394, 0.00011248740370519313, -9.790135178118369e-05],[0.432265829105921, -0.258711760783349, 0.2938331774166932], [-0.7070103500224592, 0.7072031834791394, 0.00011248740370519313, -9.790135178118369e-05],[0.432265829105921, -0.258711760783349, 0.2938331774166932], [-0.7070103500224592, 0.7072031834791394, 0.00011248740370519313, -9.790135178118369e-05],[0.4324555909394804, -0.2588315903515503, 0.17411472532856895], [-0.7070103430970139, 0.7072031652619044, 0.00021194716401288366, -0.00011347152034280979],[0.4324555909394804, -0.2588315903515503, 0.17411472532856895], [-0.7070103430970139, 0.7072031652619044, 0.00021194716401288366, -0.00011347152034280979],[0.4324555909394804, -0.2588315903515503, 0.17411472532856895], [-0.7070103430970139, 0.7072031652619044, 0.00021194716401288366, -0.00011347152034280979],[0.4482137829298416, -0.2732290534782973, 0.21091665347137067], [-0.7069893437544359, 0.707221614281854, 0.0005078778294051771, 0.0018434126914335031],[0.4482137829298416, -0.2732290534782973, 0.21091665347137067], [-0.7069893437544359, 0.707221614281854, 0.0005078778294051771, 0.0018434126914335031],[0.4482137829298416, -0.2732290534782973, 0.21091665347137067], [-0.7069893437544359, 0.707221614281854, 0.0005078778294051771, 0.0018434126914335031],[0.44881135986240817, -0.2746774308049141, 0.3337901836798805], [-0.7069903091241532, 0.7072222627872196, 2.8178234708752112e-05, -0.0011717630504317477],[0.44881135986240817, -0.2746774308049141, 0.3337901836798805], [-0.7069903091241532, 0.7072222627872196, 2.8178234708752112e-05, -0.0011717630504317477],[0.44881135986240817, -0.2746774308049141, 0.3337901836798805], [-0.7069903091241532, 0.7072222627872196, 2.8178234708752112e-05, -0.0011717630504317477],[0.46480604419804683, -0.22909827966381646, 0.42507500260802705], [0.7389880974813369, -0.6724244225585663, 0.0028568614573713, 0.04163923714846781],[0.46480604419804683, -0.22909827966381646, 0.42507500260802705], [0.7389880974813369, -0.6724244225585663, 0.0028568614573713, 0.04163923714846781],[0.46480604419804683, -0.22909827966381646, 0.42507500260802705], [0.7389880974813369, -0.6724244225585663, 0.0028568614573713, 0.04163923714846781],[0.5654765952831474, -0.07269242440954699, 0.428310621356436], [0.8323503061199624, -0.553473554730081, 0.008089834828734257, 0.028187704569489096],[0.5654765952831474, -0.07269242440954699, 0.428310621356436], [0.8323503061199624, -0.553473554730081, 0.008089834828734257, 0.028187704569489096],[0.5654765952831474, -0.07269242440954699, 0.428310621356436], [0.8323503061199624, -0.553473554730081, 0.008089834828734257, 0.028187704569489096],[0.6076516458538747, 0.12709223284949006, 0.4250012844931253], [0.9063454911413193, -0.42246045880468697, 0.003910439664156443, 0.007051233648593142],[0.6076516458538747, 0.12709223284949006, 0.4250012844931253], [0.9063454911413193, -0.42246045880468697, 0.003910439664156443, 0.007051233648593142],[0.6076516458538747, 0.12709223284949006, 0.4250012844931253], [0.9063454911413193, -0.42246045880468697, 0.003910439664156443, 0.007051233648593142],[0.607253323038377, 0.17286111939973686, 0.3815136376679544], [0.9191030537511439, -0.3940162700912953, -0.0003747625578306913, -0.0007842459376078606],[0.607253323038377, 0.17286111939973686, 0.3815136376679544], [0.9191030537511439, -0.3940162700912953, -0.0003747625578306913, -0.0007842459376078606],[0.607253323038377, 0.17286111939973686, 0.3815136376679544], [0.9191030537511439, -0.3940162700912953, -0.0003747625578306913, -0.0007842459376078606],[0.607684924736981, 0.17279763435515705, 0.2558840462339237], [0.9191468386266584, -0.39391506977989277, -3.715805917683934e-05, 7.390670249926208e-05],[0.607684924736981, 0.17279763435515705, 0.2558840462339237], [0.9191468386266584, -0.39391506977989277, -3.715805917683934e-05, 7.390670249926208e-05],[0.607684924736981, 0.17279763435515705, 0.2558840462339237], [0.9191468386266584, -0.39391506977989277, -3.715805917683934e-05, 7.390670249926208e-05],[0.6141871921233992, 0.09909166257192689, 0.21647671166078825], [0.8944972199772878, -0.44589357643171806, -0.014132002056328625, -0.029222054474575196],[0.6141871921233992, 0.09909166257192689, 0.21647671166078825], [0.8944972199772878, -0.44589357643171806, -0.014132002056328625, -0.029222054474575196],[0.6141871921233992, 0.09909166257192689, 0.21647671166078825], [0.8944972199772878, -0.44589357643171806, -0.014132002056328625, -0.029222054474575196],[0.5778304366974453, -0.08143687827171751, 0.30877808002998786], [0.8264962999641083, -0.5627609612342237, -0.004037338965199339, -0.013699144150323654],[0.5778304366974453, -0.08143687827171751, 0.30877808002998786], [0.8264962999641083, -0.5627609612342237, -0.004037338965199339, -0.013699144150323654],[0.5778304366974453, -0.08143687827171751, 0.30877808002998786], [0.8264962999641083, -0.5627609612342237, -0.004037338965199339, -0.013699144150323654],[0.48008988186598106, -0.22118099912075934, 0.3944599858157116], [0.7429420218629544, -0.6693557605123946, -9.933020797153274e-05, -9.029157784537373e-05],[0.48008988186598106, -0.22118099912075934, 0.3944599858157116], [0.7429420218629544, -0.6693557605123946, -9.933020797153274e-05, -9.029157784537373e-05],[0.48008988186598106, -0.22118099912075934, 0.3944599858157116], [0.7429420218629544, -0.6693557605123946, -9.933020797153274e-05, -9.029157784537373e-05]
'''
