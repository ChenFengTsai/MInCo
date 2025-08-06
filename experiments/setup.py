import argparse
import glob
import json
import numpy as np
import os
import pathlib
import pipes
import random
import sys
import torch

import wandb

from common.logger import configure_logger
from common.utils import set_gpu_mode

sys.path.append(".")
os.environ["WANDB_START_METHOD"] = "thread"


class AttrDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def arg_type(value):
    if isinstance(value, bool):
        return lambda x: bool(["False", "True"].index(x))
    if isinstance(value, int):
        return lambda x: float(x) if ("e" in x or "." in x) else int(x)
    if isinstance(value, pathlib.Path):
        return lambda x: pathlib.Path(x).expanduser()
    return type(value)


def parse_arguments(config):
    parser = argparse.ArgumentParser()
    for key, value in config.items():
        parser.add_argument(f"--{key}", type=arg_type(value), default=value)
    return parser.parse_args()


def get_latest_run_id(path):
    max_run_id = 0
    for path in glob.glob(os.path.join(path, "[0-9]*")):
        id = path.split(os.sep)[-1]
        if id.isdigit() and int(id) > max_run_id:
            max_run_id = int(id)
    return max_run_id


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def set_device(use_gpu, gpu_id=0):
    set_gpu_mode(use_gpu, gpu_id)


def save_cmd(base_dir):
    cmd_path = os.path.join(base_dir, "cmd.txt")
    cmd = "python " + " ".join([sys.argv[0]] + [pipes.quote(s) for s in sys.argv[1:]])
    cmd += "\n"
    print("\n" + "*" * 80)
    print("Training command:\n" + cmd)
    print("*" * 80 + "\n")
    with open(cmd_path, "w") as f:
        f.write(cmd)


def save_git(base_dir):
    git_path = os.path.join(base_dir, "git.txt")
    print("Save git commit and diff to {}".format(git_path))
    cmds = [
        "echo `git rev-parse HEAD` > {}".format(git_path),
        "git diff >> {}".format(git_path),
    ]
    os.system("\n".join(cmds))


def save_cfg(base_dir, cfg):
    cfg_path = os.path.join(base_dir, "cfg.json")
    print("Save config to {}".format(cfg_path))
    cfg_dict = vars(cfg).copy()
    for key, val in cfg_dict.items():
        if isinstance(val, pathlib.PosixPath):
            cfg_dict[key] = str(val)
    with open(cfg_path, "w") as f:
        json.dump(cfg_dict, f, indent=4)

def setup_logger(config):
    # Use the configurable base log path
    base_log_path = config.logdir
    
    print(f"Setting up logger with base path: {base_log_path}")
    
    # Configure logger directory
    logdir = os.path.join(
        base_log_path,
        config.env_id,
        config.algo,
        config.expr_name,
        str(config.seed),
    )
    
    # Create wandb directory structure
    wandb_dir = os.path.join(
        base_log_path,
        config.env_id,
        config.algo,
        config.expr_name,
        str(config.seed),
        "wandb_logs",
    )
    
    # Create wandb cache directory
    wandb_cache_dir = os.path.join(base_log_path, "wandb_cache")
    
    print(f"Creating directories:")
    print(f"  logdir: {logdir}")
    print(f"  wandb_dir: {wandb_dir}")
    print(f"  wandb_cache_dir: {wandb_cache_dir}")
    
    # Create ALL directories first
    try:
        # Create main log directory
        os.makedirs(logdir, exist_ok=True)
        print(f"✓ Created logdir: {logdir}")
        
        # Create wandb directory
        os.makedirs(wandb_dir, exist_ok=True)
        print(f"✓ Created wandb_dir: {wandb_dir}")
        
        # Create wandb cache directory
        os.makedirs(wandb_cache_dir, exist_ok=True)
        print(f"✓ Created wandb_cache_dir: {wandb_cache_dir}")
        
        # Verify directories exist and are writable
        assert os.path.exists(logdir), f"logdir doesn't exist: {logdir}"
        assert os.path.exists(wandb_dir), f"wandb_dir doesn't exist: {wandb_dir}"
        assert os.path.exists(wandb_cache_dir), f"wandb_cache_dir doesn't exist: {wandb_cache_dir}"
        
        assert os.access(logdir, os.W_OK), f"logdir not writable: {logdir}"
        assert os.access(wandb_dir, os.W_OK), f"wandb_dir not writable: {wandb_dir}"
        assert os.access(wandb_cache_dir, os.W_OK), f"wandb_cache_dir not writable: {wandb_cache_dir}"
        
        print("✓ All directories created and verified writable")
        
        # Set environment variables AFTER creating directories
        os.environ["WANDB_DIR"] = wandb_dir
        os.environ["WANDB_CACHE_DIR"] = wandb_cache_dir
        
        print(f"✓ Set WANDB_DIR to: {wandb_dir}")
        print(f"✓ Set WANDB_CACHE_DIR to: {wandb_cache_dir}")
        
    except Exception as e:
        print(f"✗ Error creating directories: {e}")
        print("Falling back to default wandb location")
        # Don't set WANDB_DIR/WANDB_CACHE_DIR, let wandb use defaults
        
    # Parse environment for wandb organization
    env_clean = config.env_id.replace('dmc_', '')
    env_parts = env_clean.split('-')
    
    if len(env_parts) == 3:
        domain_modifier = env_parts[0]  # "distracted"
        domain = env_parts[1]           # "walker"  
        task = env_parts[2]             # "walk"
        full_task = f"{domain_modifier}_{domain}_{task}"
    elif len(env_parts) == 2:
        domain = env_parts[0]           # "walker"
        task = env_parts[1]             # "walk"
        full_task = f"{domain}_{task}"
    else:
        full_task = env_parts[0]
        domain = env_parts[0]
        task = "default"
    
    # Create run name with hyperparameters
    name_parts = [f"seed{config.seed}"]
    if config.algo == "minco":
        name_parts.extend([f"a{config.a}", f"b{config.b}", f"c{config.c}"])
        if config.cross_inv_dynamics:
            name_parts.append("cross_inv")
        if config.prior_train_steps != 5:  # Only if non-default
            name_parts.append(f"prior{config.prior_train_steps}")
    name = "_".join(name_parts)
    
    # Create comprehensive tags
    tags = [
        config.algo,
        config.expr_name,
        f"domain_{domain}",
        f"task_{task}",
        f"seed_{config.seed}",
        "pixel" if config.pixel_obs else "state",
    ]
    
    # Add domain modifier if exists
    if len(env_parts) == 3:
        tags.append(f"modifier_{domain_modifier}")
    
    # Add algorithm-specific tags
    if config.algo == "minco":
        tags.extend([
            f"a_{config.a}",
            f"b_{config.b}", 
            f"c_{config.c}",
            f"prior_steps_{config.prior_train_steps}",
        ])
        if config.cross_inv_dynamics:
            tags.append("cross_inv_dynamics")
    
    # Initialize WANDB
    print("Initializing wandb...")
    wandb.init(
        project=f"dmc-{config.algo}",
        entity=os.environ.get("MY_WANDB_ID", None),
        name=name,
        group=f"{full_task}_{config.expr_name}",
        job_type=config.expr_name,
        tags=tags,
        config=config,
    )
    
    print(f"✓ Wandb initialized. Run dir: {wandb.run.dir}")
    
    # Configure logger
    logger = configure_logger(logdir, ["stdout", "tensorboard", "wandb"])
    # logger = configure_logger(logdir, ["stdout", "tensorboard"])
    config.training_log_dir = logdir

    # Log experiment info
    save_cmd(logdir)
    save_git(logdir)
    save_cfg(logdir, config)
    
    print(f"✓ Logger setup complete!")
    print(f"  Logs saved to: {logdir}")
    print(f"  Wandb saved to: {wandb.run.dir}")
    
    return logger
