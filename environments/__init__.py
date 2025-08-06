import gym
import sys
from functools import partial

sys.path.append("./environments")

from .vec_env import AsyncVecEnv
from .wrappers import (
    CastObs,
    TimeLimit,
    ActionRepeat,
    NormalizeAction,
    MetaWorldWrapper,
    FrankaWrapper,
    MazeWrapper,
)

gym.logger.set_level(40)



def make_env(env_id, seed, pixel_obs=False, **kwargs):
    suite, task = env_id.split("-", 1)
    if suite == "mw":
        from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_HIDDEN

        env = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN[f"{task}-v2-goal-hidden"]()
        env = MetaWorldWrapper(env, pixel_obs)
        env = TimeLimit(env, 150)
    elif suite[:3] == "dmc":
        if suite[:5] == "dmcgb":
            # DMCGB environments with advanced distracting control
            env = _make_dmcgb_env(task, seed, pixel_obs, **kwargs)
        else:
            
            from .dmc import DMCEnv

            img_source = None
            resource_files = None
            reset_bg = False
            if suite == "dmc_static":
                img_source = "images"
                resource_files = "../data/imagenet/*.JPEG"
            elif suite == "dmc_static_reset":
                img_source = "images"
                resource_files = "../data/imagenet/*.JPEG"
                reset_bg = True
            elif suite == "dmc_distracted":
                img_source = "video"
                resource_files = "/storage/ssd1/richtsai1103/kinetics400/*.mp4"
            
            env = DMCEnv(
                name=task,
                pixel_obs=pixel_obs,
                img_source=img_source,
                resource_files=resource_files,
                total_frames=1000,
                reset_bg=reset_bg,
            )
            env = NormalizeAction(env)
            env = TimeLimit(env, 1000)
            env = ActionRepeat(env, 2)
    elif suite == "franka":
        from .tabletop import FRANKA_ENVIRONMENTS

        env = FRANKA_ENVIRONMENTS[task]("environments/tabletop/assets")
        env = FrankaWrapper(env, pixel_obs)
        env = TimeLimit(env, 200)
    elif suite == "pointmass":
        from .tabletop import PointmassReachEnv

        env = PointmassReachEnv("environments/tabletop/assets", task, pixel_obs)
        env = TimeLimit(env, 50)
    elif suite == "maze" or suite == "maze_distracted":
        import maze

        env_kwargs = {}
        if task == "obstacle":
            env_kwargs["reset_locations"] = [(3, 1)]
        env = gym.make(f"maze2d-{task}-v0", **env_kwargs)
        env = MazeWrapper(
            env=env,
            pixel_obs=pixel_obs,
            img_source="video" if "distracted" in suite else None,
            resource_files="../kinetics-downloader/dataset/train/driving_car/*.mp4",
            total_frames=1000,
        )
        
    else:
        env = gym.make(env_id)
    
    return env


def _make_dmcgb_env(task, seed, pixel_obs, **kwargs):
    """
    Create DMCGB environment with advanced distracting control support.
    """
    action_repeat = kwargs.get('action_repeat', 2)
    image_size = kwargs.get('size', 64)
    # ds_resource_path = kwargs.get('ds_resource_path', None)
    # distracting_cs_intensity = kwargs.get('distracting_cs_intensity', 0.1)
    # frame_stack = kwargs.get('frame_stack', None)
    
    # Parse task to extract domain and actual task
    task_parts = task.split("-")
    train_mode = task_parts[2]
    
    
    if train_mode in ['color_easy', 'color_hard', 'video_easy', 'video_hard']:
        # distracting control
        task_parts = task.split("-")
        domain = task_parts[0]              # "walker"
        actual_task = task_parts[1]         # "walk"
        # Try to import and use the gb wrappers
        from env.wrappers import make_env as gb_make_env
        env = gb_make_env(
            domain_name=domain,
            task_name=actual_task, 
            seed=seed,
            episode_length=1000,
            action_repeat=2,
            image_size=image_size,
            mode=train_mode,
            # intensity=distracting_cs_intensity,
            # ds_resource_path=[ds_resource_path] if ds_resource_path else None
        )
        import envs.wrappers
        # Apply your existing DMC2GYMWrapper
        env = envs.wrappers.DMC2GYMWrapper(env)
        
        # Apply DMCGBEnvWrapper to make it compatible with MInCo
        env = envs.wrappers.DMCGBEnvWrapper(env, pixel_obs=pixel_obs, resolution=image_size)
        env = NormalizeAction(env)
        time_limit = 1000
        action_repeat = 2
        env = TimeLimit(env, time_limit // action_repeat)
            
    else:
        # Fallback to simple DMC
        from .dmc import DMCEnv
        env = DMCEnv(name=task, pixel_obs=pixel_obs)
        env = NormalizeAction(env)
        env = ActionRepeat(env, action_repeat)
    
    return env


def make_vec_env(env_id, num_envs, seed, pixel_obs=False):
    env_fns = [
        partial(make_env, env_id=env_id, seed=seed + i, pixel_obs=pixel_obs)
        for i in range(num_envs)
    ]
    return AsyncVecEnv(env_fns)


