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
        # print(prompt)
        # log_id = 0
        # if i999==1:
        #     log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "robotmove.mp4")
        '''
        import pybullet as p
        import pybullet_data
        import time 
        from pprint import pprint
        
        # 连接物理引擎
        use_gui = True
        if use_gui:
            serve_id = p.connect(p.GUI)
        else:
            serve_id = p.connect(p.DIRECT)
        
        # 添加资源路径
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # 配置渲染机制
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        
        # 设置重力，加载模型
        p.setGravity(0, 0, -10)
        _ = p.loadURDF("plane.urdf", useMaximalCoordinates=True)
        robot_id = p.loadURDF("r2d2.urdf", useMaximalCoordinates=True)
        
        available_joints_indexes = [i for i in range(p.getNumJoints(robot_id)) if
                                    p.getJointInfo(robot_id, i)[2] != p.JOINT_FIXED]
        pprint([p.getJointInfo(robot_id, i)[1] for i in available_joints_indexes])
        '''
        # robot_id = 2
        # robot_id = 10
        # print("robot_id: ", robot_id)
        # available_joints_indexes = [i for i in range(p.getNumJoints(robot_id)) if
        #                             p.getJointInfo(robot_id, i)[2] != p.JOINT_FIXED]
        # print([p.getJointInfo(robot_id, i)[1] for i in available_joints_indexes])
        # print(p.getJointInfo(robot_id, 4)[1])
        # numBodies = p.getNumBodies()
        # print("numBodies: ", numBodies)
        # for i in range (numBodies):
        #     bodyInfo = p.getBodyInfo(i)
        #     print("bodyInfo: ", i, bodyInfo)

        # bodyInfo = p.getBodyInfo(3) # base
        # print(bodyInfo, type(bodyInfo))
        # for i in bodyInfo:
        #     print(i, type(i))
        # bodyInfo = p.getBodyInfo(4) # head
        # info = p.getLinkState(3,0)
        # print("info: ", info)

        # for _ in range(task.oracle_max_steps):
        for _ in range(task.oracle_max_steps):

            # print(task.task_name, task.oracle_max_steps)
            oracle_action = oracle_fn.act(obs)
            # clamp action to valid range
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

            # print("info: ", info)
            # print("oracle_action: \n", oracle_action)

            obs, reward, done, info = env.step(action=oracle_action, skip_oracle=False, episode=env.episode_counter, task_name=task.task_name)
            if done:
                break
            import time
            # time.sleep(1)
        # env.episode_counter += 1
        print("episode_counter: ", env.episode_counter)
        # print("i999: ", i999)
        # if i999==5:
        #     p.stopStateLogging(log_id)

        seed += 1


if __name__ == "__main__":
    main()
