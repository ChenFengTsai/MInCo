import datetime
import gym
import numpy as np
import uuid
from gym import spaces

class TimeLimit:
    def __init__(self, env, duration):
        self._env = env
        self._duration = duration
        self._step = None

    def __getattr__(self, name):
        return getattr(self._env, name)
    
    def step(self, action):
        assert self._step is not None, "Must reset environment."
        obs, reward, done, info = self._env.step(action)
        self._step += 1
        if self._step >= self._duration:
            done = True
            if "discount" not in info:
                info["discount"] = np.array(1.0).astype(np.float32)
            self._step = None
        return obs, reward, done, info

    def reset(self):
        self._step = 0
        return self._env.reset()


class NormalizeActions:
    def __init__(self, env):
        self._env = env
        self._mask = np.logical_and(
            np.isfinite(env.action_space.low), np.isfinite(env.action_space.high)
        )
        self._low = np.where(self._mask, env.action_space.low, -1)
        self._high = np.where(self._mask, env.action_space.high, 1)
        low = np.where(self._mask, -np.ones_like(self._low), self._low)
        high = np.where(self._mask, np.ones_like(self._low), self._high)
        self.action_space = gym.spaces.Box(low, high, dtype=np.float32)

    def __getattr__(self, name):
        return getattr(self._env, name)
    
    def step(self, action):
        original = (action + 1) / 2 * (self._high - self._low) + self._low
        original = np.where(self._mask, original, action)
        return self._env.step(original)


class OneHotAction:
    def __init__(self, env):
        assert isinstance(env.action_space, gym.spaces.Discrete)
        self._env = env
        self._random = np.random.RandomState()
        shape = (self._env.action_space.n,)
        space = gym.spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)
        space.discrete = True
        self.action_space = space

    def __getattr__(self, name):
        return getattr(self._env, name)
    
    def step(self, action):
        index = np.argmax(action).astype(int)
        reference = np.zeros_like(action)
        reference[index] = 1
        if not np.allclose(reference, action):
            raise ValueError(f"Invalid one-hot action:\n{action}")
        return self._env.step(index)

    def reset(self):
        return self._env.reset()

    def _sample_action(self):
        actions = self._env.action_space.n
        index = self._random.randint(0, actions)
        reference = np.zeros(actions, dtype=np.float32)
        reference[index] = 1.0
        return reference


class RewardObs:
    def __init__(self, env):
        self._env = env
        spaces = self._env.observation_space.spaces
        if "obs_reward" not in spaces:
            spaces["obs_reward"] = gym.spaces.Box(
                -np.inf, np.inf, shape=(1,), dtype=np.float32
            )
        self.observation_space = gym.spaces.Dict(spaces)

    def __getattr__(self, name):
        return getattr(self._env, name)
    
    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        if "obs_reward" not in obs:
            obs["obs_reward"] = np.array([reward], dtype=np.float32)
        return obs, reward, done, info

    def reset(self):
        obs = self._env.reset()
        if "obs_reward" not in obs:
            obs["obs_reward"] = np.array([0.0], dtype=np.float32)
        return obs


class SelectAction:
    def __init__(self, env, key):
        self._env = env
        self._key = key

    def __getattr__(self, name):
        return getattr(self._env, name)
    
    def step(self, action):
        return self._env.step(action[self._key])


class UUID:
    def __init__(self, env):
        self._env = env
        timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        self.id = f"{timestamp}-{str(uuid.uuid4().hex)}"

    def __getattr__(self, name):
        return getattr(self._env, name)
    
    def reset(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        self.id = f"{timestamp}-{str(uuid.uuid4().hex)}"
        return self._env.reset()
    
class ClipAction:

    def __init__(self, env, key='action', low=-1, high=1):
        self._env = env
        self._key = key
        self._low = low
        self._high = high

    def __getattr__(self, name):
        return getattr(self._env, name)
    
    def step(self, action):
        clipped = np.clip(action[self._key], self._low, self._high)
        return self._env.step({**action, self._key: clipped})

class DMC2GYMWrapper:

    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        assert np.isfinite(action).all(), action
        time_step, reward, done, info = self._env.step(action)
        obs = dict(time_step.observation)
        obs = {key: [val] if len(val.shape) == 0 else val for key, val in obs.items()}
        # obs["image"] = self.render()
        # There is no terminal state in DMC
        obs["is_terminal"] = False if time_step.first() else time_step.discount == 0
        obs["is_first"] = time_step.first()
        done = time_step.last()
        info = {"discount": np.array(time_step.discount, np.float32)}
        return obs, reward, done, info

    def reset(self):
        time_step = self._env.reset()
        obs = dict(time_step.observation)
        obs = {key: [val] if len(val.shape) == 0 else val for key, val in obs.items()}
        # obs["image"] = self.render()
        obs["is_terminal"] = False if time_step.first() else time_step.discount == 0
        obs["is_first"] = time_step.first()
        return obs


class DMCGBEnvWrapper(gym.Env):
    """
    Wrapper that makes DMCGB environments compatible with MInCo by providing
    proper observation space and converting dict observations to arrays.
    This mimics the structure of your existing DMCEnv.
    """
    def __init__(self, dmcgb_env, pixel_obs=True, resolution=64):
        self._env = dmcgb_env
        self._pixel_obs = pixel_obs
        self._resolution = resolution
        
        # Setup observation space similar to your DMCEnv
        if pixel_obs:
            img_shape = (3, self._resolution, self._resolution)
            self.observation_space = spaces.Box(
                low=0, high=255, shape=img_shape, dtype=np.uint8
            )
        else:
            # Determine state observation size from the environment
            obs_len = self._get_state_obs_length()
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_len,), dtype=np.float32
            )
        
        # Copy action space from the wrapped environment
        self.action_space = dmcgb_env.action_space
    
    def _get_state_obs_length(self):
        """Determine the length of state observations by sampling."""
        try:
            sample_obs = self._env.reset()
            state_parts = []
            for key, value in sample_obs.items():
                if key not in ['image', 'is_terminal', 'is_first']:
                    if isinstance(value, (list, tuple)):
                        state_parts.append(len(value))
                    elif np.isscalar(value):
                        state_parts.append(1)
                    else:
                        state_parts.append(np.prod(value.shape))
            return sum(state_parts) if state_parts else 24
        except:
            return 24  # Default fallback
    
    def seed(self, seed):
        self.observation_space.seed(seed)
        self.action_space.seed(seed)
        if hasattr(self._env, 'seed'):
            self._env.seed(seed)
    
    def step(self, action):
        obs_dict, reward, done, info = self._env.step(action)
        obs = self._process_observation(obs_dict)
        return obs, reward, done, info
    
    def reset(self):
        obs_dict = self._env.reset()
        obs = self._process_observation(obs_dict)
        return obs
    
    def render(self, mode="rgb_array", height=64, width=64, camera_id=0):
        if hasattr(self._env, 'render'):
            return self._env.render(mode=mode, height=height, width=width, camera_id=camera_id)
        else:
            # Fallback for environments without render method
            return np.zeros((height, width, 3), dtype=np.uint8)
    
    def _process_observation(self, obs_dict):
        """Convert dict observation to the format expected by MInCo."""
        if self._pixel_obs:
            # Extract image observation
            if 'image' in obs_dict:
                image = obs_dict['image']
                # Ensure the image is in CHW format like your DMCEnv
                if image.ndim == 3:
                    if image.shape[-1] == 3:  # HWC -> CHW
                        image = np.transpose(image, (2, 0, 1))
                return image.astype(np.uint8)
            else:
                # Fallback: create dummy image
                return np.zeros((3, self._resolution, self._resolution), dtype=np.uint8)
        else:
            # Extract state observations (similar to _flatten_obs in your DMCEnv)
            state_parts = []
            for key, value in obs_dict.items():
                if key not in ['image', 'is_terminal', 'is_first']:
                    if isinstance(value, (list, tuple)):
                        value = np.array(value)
                    if np.isscalar(value):
                        state_parts.append(np.array([value]))
                    else:
                        state_parts.append(value.flatten())
            
            if state_parts:
                state = np.concatenate(state_parts, axis=0)
                return state.astype(np.float32)
            else:
                # Fallback: create dummy state
                return np.zeros(self.observation_space.shape, dtype=np.float32)
            
            
class SparseReward:
    def __init__(self, env):
        self._env = env

    def step(self, action):
        obs, _, done, info = self._env.step(action)
        reward = float(info["success"])
        return obs, reward, done, info