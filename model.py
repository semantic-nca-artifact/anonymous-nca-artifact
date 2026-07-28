import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CHANNEL_N, CELL_FIRE_RATE


def get_living_mask(x: torch.Tensor) -> torch.Tensor:
    """Return cells adjacent to positive raw-alpha state."""
    alpha = x[:, 3:4].clamp(0.0, 1.0)
    return F.max_pool2d(alpha, 3, stride=1, padding=1) > 0.1


def make_seed(size: int, n: int = 1) -> np.ndarray:
    """Construct a batch of single-cell NCHW initial states."""
    x = np.zeros([n, CHANNEL_N, size, size], np.float32)
    x[:, 3:, size // 2, size // 2] = 1.0
    return x


class CAModel(nn.Module):

    def __init__(self, channel_n: int = CHANNEL_N, fire_rate: float = CELL_FIRE_RATE):
        super().__init__()
        self.channel_n = channel_n
        self.fire_rate = fire_rate

        identify = np.float32([0, 1, 0])
        identify = np.outer(identify, identify)
        dx = np.outer([1, 2, 1], [-1, 0, 1]) / 8.0
        dy = dx.T
        kernels = np.stack([identify, dx, dy], axis=0)
        kernels = np.tile(kernels[:, None], (1, channel_n, 1, 1))
        zero_angle_kernel = torch.tensor(
            kernels.transpose(1, 0, 2, 3).reshape(channel_n * 3, 1, 3, 3),
            dtype=torch.float32,
        )
        self.register_buffer(
            '_zero_angle_perception_kernel', zero_angle_kernel, persistent=False
        )

        self.dmodel = nn.Sequential(
            nn.Conv2d(channel_n * 3, 128, 1),
            nn.ReLU(),
            nn.Conv2d(128, channel_n, 1),
        )
        nn.init.zeros_(self.dmodel[-1].weight)
        nn.init.zeros_(self.dmodel[-1].bias)

    def perceive(self, x: torch.Tensor, angle: float = 0.0) -> torch.Tensor:
        if float(angle) == 0.0:
            kernel = self._zero_angle_perception_kernel.to(dtype=x.dtype)
            return F.conv2d(x, kernel, padding=1, groups=self.channel_n)

        identify = np.float32([0, 1, 0])
        identify = np.outer(identify, identify)
        dx = np.outer([1, 2, 1], [-1, 0, 1]) / 8.0
        dy = dx.T
        c, s = np.cos(angle), np.sin(angle)
        # Repeat the three perception kernels for depthwise convolution.
        kernels = np.stack([identify, c * dx - s * dy, s * dx + c * dy], axis=0)  # [3, 3, 3]
        kernels = np.tile(kernels[:, None], (1, self.channel_n, 1, 1))  # [3, C, 3, 3]
        # reshape for grouped depthwise: [C*3, 1, 3, 3]
        k = torch.tensor(
            kernels.transpose(1, 0, 2, 3).reshape(self.channel_n * 3, 1, 3, 3),
            dtype=x.dtype, device=x.device,
        )
        return F.conv2d(x, k, padding=1, groups=self.channel_n)

    def forward(
        self,
        x: torch.Tensor,
        fire_rate: float = None,
        angle: float = 0.0,
        step_size: float = 1.0,
        return_premask: bool = False,
    ) -> torch.Tensor:
        pre_life_mask = get_living_mask(x)

        y = self.perceive(x, angle)
        dx = self.dmodel(y) * step_size

        if fire_rate is None:
            fire_rate = self.fire_rate
        update_mask = torch.rand_like(x[:, :1]) <= fire_rate
        x_pre_mask = x + dx * update_mask.float()

        post_life_mask = get_living_mask(x_pre_mask)
        life_mask = (pre_life_mask & post_life_mask).float()
        x_out = x_pre_mask * life_mask

        if return_premask:
            return x_out, x_pre_mask
        return x_out
