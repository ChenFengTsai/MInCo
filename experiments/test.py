import os
import sys
import numpy as np
import torch
import argparse
from copy import deepcopy

# Add the minco path
sys.path.append("/home/richtsai1103/CRL/minco")

from minco import MInCo
from environments import make_env
from setup import AttrDict, set_seed, set_device
from common.utils import to_torch, to_np, preprocess

# Set mujoco rendering
os.environ['MUJOCO_GL'] = 'egl'


def load_checkpoint_safe(agent, params):
    """Safely load checkpoint parameters with error handling for config mismatches"""
    agent.step = params["step"]
    agent.encoder.load_state_dict(params["encoder"])
    agent.transition_model.load_state_dict(params["transition_model"])
    agent.obs_model.load_state_dict(params["obs_model"])
    agent.reward_model.load_state_dict(params["reward_model"])
    agent.actor_model.load_state_dict(params["actor_model"])
    agent.value_model.load_state_dict(params["value_model"])
    agent.model_optimizer.load_state_dict(params["model_optimizer"])
    agent.actor_optimizer.load_state_dict(params["actor_optimizer"])
    agent.value_optimizer.load_state_dict(params["value_optimizer"])

    # Load simsiam if it exists
    if hasattr(agent, 'simsiam') and "simsiam" in params:
        agent.simsiam.load_state_dict(params["simsiam"])

    # Load disagreement model if configured and exists in checkpoint
    if hasattr(agent.c, 'disag_model') and agent.c.disag_model and "disag_model" in params:
        agent.disag_model.load_state_dict(params["disag_model"])
        agent.disag_optimizer.load_state_dict(params["disag_optimizer"])

    # Load inverse dynamics model if configured and exists in checkpoint
    if hasattr(agent.c, 'cross_inv_dynamics') and agent.c.cross_inv_dynamics and "inv_dynamics" in params:
        agent.inv_dynamics.load_state_dict(params["inv_dynamics"])
        if hasattr(agent, 'inv_dynamics_optimizer') and "inv_dynamics_optimizer" in params:
            agent.inv_dynamics_optimizer.load_state_dict(params["inv_dynamics_optimizer"])


def load_config_from_checkpoint(checkpoint_dir):
    """Load configuration from checkpoint directory"""
    import json
    config_path = os.path.join(checkpoint_dir, 'config.json')
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        config = AttrDict(config_dict)
    else:
        # Fallback to default config if config.json not found
        print(f"Warning: config.json not found in {checkpoint_dir}, using default config")
        config = get_default_config()
    
    return config


def get_default_config():
    """Default configuration matching your training setup"""
    config = AttrDict()
    config.algo = "minco"
    config.env_id = "dmcgb-walker-walk-video_hard"
    config.seed = 0
    config.use_gpu = True
    config.gpu_id = 1
    config.size = [64, 64]

    # Dreamer parameters
    config.pixel_obs = True
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
    config.load_checkpoint = True
    config.save_buffer = False

    # MInCo specific
    config.a = 8e-6
    config.b = 5
    config.c = 0.007
    config.cross_inv_dynamics = False
    config.inv_dynamics_hidden_size = 512

    # Disagreement model
    config.disag_model = False
    config.ensemble_size = 6
    config.disag_lr = 3e-4
    config.disag_coef = 0.0

    # RePo
    config.target_kl = 3.0
    config.beta_lr = 1e-4
    config.init_beta = 1e-5
    config.prior_train_steps = 5
    config.kl_balance = 0.8

    return config


class DummyLogger:
    """Dummy logger for evaluation"""
    def __init__(self):
        self.dir = None
    
    def record(self, key, value, exclude=None):
        pass
    
    def dump(self, step):
        pass


def evaluate_agent(agent, eval_env, num_episodes=100, max_episode_steps=1000):
    """Evaluate agent for specified number of episodes with different backgrounds"""
    
    agent.toggle_train(False)  # Set to evaluation mode
    
    episode_returns = []
    episode_successes = []
    episode_lengths = []
    
    print(f"Starting evaluation for {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        # Reset environment with new background
        obs = eval_env.reset()
        
        # Initialize agent state
        belief, posterior_state, action_tensor = agent.init_latent_and_action()
        
        episode_reward = 0
        episode_success = 0
        episode_length = 0
        done = False
        
        with torch.no_grad():
            while not done and episode_length < max_episode_steps:
                # Preprocess observation
                obs_tensor = to_torch(preprocess(obs[None]))
                
                # Get action from agent
                (
                    belief,
                    posterior_state,
                    action_tensor,
                ) = agent.update_latent_and_select_action(
                    belief, posterior_state, action_tensor, obs_tensor, explore=False
                )
                
                # Convert action to numpy
                action = to_np(action_tensor)[0]
                
                # Step environment
                next_obs, reward, done, info = eval_env.step(action)
                
                # Update episode statistics
                obs = next_obs
                episode_reward += reward
                episode_success += info.get("success", 0)
                episode_length += 1
        
        # Store episode results
        episode_returns.append(episode_reward)
        episode_successes.append(float(episode_success > 0))
        episode_lengths.append(episode_length)
        
        # Print individual episode results
        print(f"Episode {episode + 1}/{num_episodes} - "
              f"Reward: {episode_reward:.3f}, "
              f"Success: {int(episode_success > 0)}, "
              f"Length: {episode_length}")
        
        # Print running statistics every 10 episodes
        if (episode + 1) % 10 == 0:
            avg_return = np.mean(episode_returns[-10:])
            avg_success = np.mean(episode_successes[-10:])
            print(f"  → Last 10 episodes avg: Return={avg_return:.3f}, Success={avg_success:.3f}")
            print()
    
    # Calculate final statistics
    results = {
        'mean_return': np.mean(episode_returns),
        'std_return': np.std(episode_returns),
        'mean_success': np.mean(episode_successes),
        'mean_length': np.mean(episode_lengths),
        'all_returns': episode_returns,
        'all_successes': episode_successes,
        'all_lengths': episode_lengths
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate MInCo agent on DMCGB video hard')
    parser.add_argument('--load_config', type=str, required=True,
                        help='Path to config JSON file')
    parser.add_argument('--load_checkpoint', type=str, required=True,
                        help='Path to checkpoint models.pt file')
    parser.add_argument('--num_eval_episodes', type=int, default=100,
                        help='Number of evaluation episodes')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for evaluation')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU ID to use')
    parser.add_argument('--max_episode_steps', type=int, default=1000,
                        help='Maximum steps per episode')
    parser.add_argument('--save_results', type=str, default=None,
                        help='Path to save evaluation results (optional)')
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Set device
    set_device(True, args.gpu_id)
    
    # Load configuration
    import json
    print(f"Loading config from: {args.load_config}")
    with open(args.load_config, 'r') as f:
        config_dict = json.load(f)
    config = AttrDict(config_dict)
    
    # Override environment to dmcgb video hard
    config.env_id = "dmcgb-walker-walk-video_hard"
    config.load_checkpoint = True
    
    print(f"Using config: {config}")
    print(f"Evaluating on environment: {config.env_id}")
    
    # Create dummy logger
    logger = DummyLogger()
    logger.dir = os.path.dirname(args.load_checkpoint)
    
    # Create environment
    eval_env = make_env(config.env_id, args.seed, config.pixel_obs)
    
    # Create a dummy training environment (required for agent initialization)
    dummy_env = make_env(config.env_id, args.seed, config.pixel_obs)
    
    print("Creating agent...")
    
    # Create agent
    agent = MInCo(config, dummy_env, eval_env, logger)
    
    # Load checkpoint
    if not os.path.exists(args.load_checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.load_checkpoint}")
    
    print(f"Loading checkpoint from: {args.load_checkpoint}")
    params = torch.load(args.load_checkpoint, map_location=agent.device)
    
    # Fix for loading params - use custom load method to handle config mismatch
    load_checkpoint_safe(agent, params)
    print(f"Loaded checkpoint from step: {params['step']}")
    
    # Run evaluation
    print(f"\nStarting evaluation with {args.num_eval_episodes} episodes...")
    results = evaluate_agent(
        agent, 
        eval_env, 
        num_episodes=args.num_eval_episodes,
        max_episode_steps=args.max_episode_steps
    )
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Environment: {config.env_id}")
    print(f"Number of episodes: {args.num_eval_episodes}")
    print(f"Checkpoint step: {params['step']}")
    print(f"Mean return: {results['mean_return']:.3f} ± {results['std_return']:.3f}")
    print(f"Mean success rate: {results['mean_success']:.3f}")
    print(f"Mean episode length: {results['mean_length']:.1f}")
    print(f"Min return: {min(results['all_returns']):.3f}")
    print(f"Max return: {max(results['all_returns']):.3f}")
    
    # Save results if requested
    if args.save_results:
        import json
        results_to_save = {
            'config_path': args.load_config,
            'checkpoint_path': args.load_checkpoint,
            'config': dict(config),
            'checkpoint_step': int(params['step']),
            'evaluation_args': vars(args),
            'results': {k: v if k.startswith('all_') else float(v) 
                       for k, v in results.items()}
        }
        
        with open(args.save_results, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        print(f"\nResults saved to: {args.save_results}")


if __name__ == "__main__":
    main()
    
# python experiments/test.py --load_config /storage/ssd1/richtsai1103/iso_ted/log/dmcgb-walker-walk-video_hard/minco/benchmark/1/cfg.json --load_checkpoint /storage/ssd1/richtsai1103/iso_ted/log/dmcgb-walker-walk-video_hard/minco/benchmark/1/models.pt --num_eval_episodes 100