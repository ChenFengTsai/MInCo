import numpy as np
import torch

import torch.nn.functional as F
from torch.distributions import Uniform

_GPU_ID = 0
_USE_GPU = True
_DEVICE = None


def set_gpu_mode(mode, gpu_id=0, use_gpu=True):
    global _GPU_ID
    global _USE_GPU
    global _DEVICE
    _GPU_ID = gpu_id
    _USE_GPU = mode
    _DEVICE = torch.device(("cuda:" + str(_GPU_ID)) if _USE_GPU else "cpu")
    
    current_device = torch.cuda.current_device()
    if _USE_GPU:
        torch.cuda.set_device(gpu_id)  # This is what was missing!
        
    current_device = torch.cuda.current_device()
        
    torch.set_default_tensor_type(
        torch.cuda.FloatTensor if _USE_GPU else torch.FloatTensor
    )


def get_device():
    global _DEVICE
    return _DEVICE


def to_torch(x, dtype=None, device=None):
    if device is None:
        device = get_device()
    return torch.as_tensor(x, dtype=dtype, device=device)


def to_np(x):
    return x.detach().cpu().numpy()


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class FreezeParameters:
    def __init__(self, params):
        self.params = params
        self.param_states = [p.requires_grad for p in self.params]

    def __enter__(self):
        for param in self.params:
            param.requires_grad = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        for i, param in enumerate(self.params):
            param.requires_grad = self.param_states[i]


def lambda_return(rewards, values, discounts, bootstrap, lambda_=0.95):
    next_values = torch.cat([values[1:], bootstrap[None]], 0)
    inputs = rewards + discounts * next_values * (1 - lambda_)
    last = bootstrap
    outputs = []
    for t in reversed(range(len(inputs))):
        last = inputs[t] + discounts[t] * lambda_ * last
        outputs.append(last)
    outputs = list(reversed(outputs))
    returns = torch.stack(outputs, 0)
    return returns


def preprocess(obs, aug=False):
    # Preprocess a batch of observations
    ndims = len(obs.shape)
    assert ndims == 2 or ndims == 4, "preprocess accepts a batch of observations"
    if ndims == 4:
        obs = ((obs.astype(np.float32) / 255) * 2) - 1.0
    
    # data augmentation
    if aug:
    
        obs = random_translate(
            obs,
            # self._config.max_delta,
            # self._config.same_across_time,
            # self._config.bilinear
            )
    
    return obs


def postprocess(obs):
    # Postprocess a batch of observations
    ndims = len(obs.shape)
    assert ndims == 2 or ndims == 4, "postprocess accepts a batch of observations"
    if ndims == 4:
        obs = np.floor((obs + 1.0) / 2 * 255).clip(0, 255).astype(np.uint8)
    return obs


def random_translate(images, max_delta=0.05, same_across_time=True, bilinear=True):
    
    shape = images.shape
    assert len(shape) == 4 and shape[0] == 2500
    B, T = 50,50

    # Convert images to PyTorch tensor
    device = get_device()  # Use the global device
    images = torch.tensor(images, device=device)
    
    
    if same_across_time:
        delta = Uniform(-max_delta, max_delta).sample([B, 1, 2]).to(images.device)
        delta = delta.repeat(1, T, 1)
    else:
        delta = Uniform(-max_delta, max_delta).sample([B, T, 2]).to(images.device)

    # Flatten the images tensor
    images_flat = images.view(B * T, *shape[1:])  # 
    
    if bilinear:
        translated_images_flat = F.grid_sample(
            images_flat,
            F.affine_grid(
                torch.eye(2, 3).unsqueeze(0).repeat(B * T, 1, 1).to(images.device)
                + torch.cat([torch.zeros_like(delta), torch.zeros_like(delta), delta], dim=-1).view(-1, 3, 2).permute(0, 2,1),
                images_flat.size(),
            ),
            mode='bilinear',
            padding_mode='border',
        )
    else:
        # Nearest-neighbor interpolation
        delta = torch.floor(delta)
        translated_images_flat = F.grid_sample(
            images_flat,
            F.affine_grid(
                torch.eye(2, 3).unsqueeze(0).repeat(B * T, 1, 1).to(images.device)
                + torch.cat([torch.zeros_like(delta), torch.zeros_like(delta), delta], dim=-1).view(-1, 3, 2).permute(0,2,1),
                images_flat.size(),
            ),
            mode='nearest',
            padding_mode='border',
        )

    # Reshape back to the original shape
    # translated_images = translated_images_flat.view(shape)
    translated_images = translated_images_flat.reshape(B, T, *shape[1:])

    return translated_images
