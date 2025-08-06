from copy import deepcopy

import sys
sys.path.append("/home/richtsai1103/CRL/minco")

from minco import Dreamer, RePo, MInCo  
from environments import make_env
from setup import AttrDict, parse_arguments, set_seed, set_device, setup_logger

import os
os.environ['MUJOCO_GL'] = 'egl'


def get_config():
    config = AttrDict()
    config.algo = "minco"
    config.env_id = "dmc_distracted-walker-walk"
    config.expr_name = "default"
    config.seed = 0
    config.use_gpu = True
    config.gpu_id = 1
    config.logdir = "/storage/ssd1/richtsai1103/iso_ted/log"  # Default value
    config.size = [64, 64]
    config.use_wandb = True
    
    # TED
    config.use_ted = False                    # Enable/disable TED
    config.ted_coefficient_start = 0.0       # Starting coefficient
    config.ted_coefficient_end = 0.1         # Final coefficient  
    config.ted_warmup_ratio = 0.2            # Warmup as fraction of total steps
    config.use_target_encoder = False        # Use target encoder or not
    config.target_tau = 0.01                 # Target encoder update rate

    # Dreamer
    config.pixel_obs = True
    config.action_repeat = 2
    # dmc_video_hard
    config.num_steps = 500000 // config.action_repeat
    
    # original
    # config.num_steps = 500000
    
    config.replay_size = 500000
    config.prefill = 2500
    config.train_every = 500
    config.train_steps = 100
    config.eval_every = 5000
    config.checkpoint_every = 25000
    config.log_every = 500
    config.embedding_size = 1024
    config.hidden_size = 200
    config.belief_size = 200
    config.state_size = 30
    config.dense_activation_function = "elu"
    config.cnn_activation_function = "relu"
    config.batch_size = 50
    config.chunk_size = 50
    config.horizon = 15
    config.gamma = 0.99
    config.gae_lambda = 0.95
    config.action_noise = 0.0
    config.action_ent_coef = 3e-4
    config.latent_ent_coef = 0.0
    config.free_nats = 3
    config.model_lr = 3e-4
    config.actor_lr = 8e-5
    config.value_lr = 8e-5
    config.grad_clip_norm = 100.0
    config.load_checkpoint = False
    config.load_offline = False
    config.offline_dir = "data"
    config.offline_truncate_size = 1000000
    config.save_buffer = False

    config.kl_balance = 0.8

    # RePo
    config.target_kl = 3.0
    config.beta_lr = 1e-4
    config.init_beta = 1e-5
    config.prior_train_steps = 5

    # Disagreement model
    config.disag_model = False
    config.ensemble_size = 6
    config.disag_lr = 3e-4
    config.disag_coef = 0.0


    # MInCo
    # if step is actor step, a = 8e-6, if step is environmental step a = 4e-6.(action repeat=2, actor step=2*environmental step)
    config.a = 8e-6
    config.b = 5
    config.c=0.007
    config.cross_inv_dynamics = False
    config.inv_dynamics_hidden_size = 512
    return parse_arguments(config)


if __name__ == "__main__":
    config = get_config()
    set_seed(config.seed)

    # cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    set_device(config.use_gpu, config.gpu_id)
    # cuda_id = cuda_visible_devices.split(',')[0]
    # set_device(config.use_gpu)

    # Logger
    logger = setup_logger(config)

    # Environment
    env = make_env(config.env_id, config.seed, config.pixel_obs)
    eval_env = make_env(config.env_id, config.seed, config.pixel_obs)

    # Sync video distractors
    if getattr(eval_env.unwrapped, "_img_source", None) is not None:
        eval_env.unwrapped._bg_source = deepcopy(env.unwrapped._bg_source)

    # Agent
    if config.algo == "dreamer":
        algo = Dreamer(config, env, eval_env, logger)
    elif config.algo == "repo":
        algo = RePo(config, env, eval_env, logger)
    elif config.algo == "minco":
        algo = MInCo(config, env, eval_env, logger)
    else:
        raise NotImplementedError("Unsupported algorithm")
    algo.train()
    # pkill -f wandb
    # ps aux | grep wandb
