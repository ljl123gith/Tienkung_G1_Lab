# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# with additional modifications by the TienKung-Lab Project,
# and is distributed under the BSD-3-Clause license.

import torch
import torch.nn as nn
from torch import autograd


class Discriminator(nn.Module):
    """
    Discriminator neural network for adversarial motion priors (AMP) reward prediction.

    Args:
        input_dim (int): Dimension of the input feature vector (concatenated state and next state).
        amp_reward_coef (float): Coefficient to scale the AMP reward.
        hidden_layer_sizes (list[int]): Sizes of hidden layers in the MLP trunk.
        device (torch.device): Device to run the model on (CPU or GPU).
        task_reward_lerp (float, optional): Interpolation factor between AMP reward and task reward.
            Defaults to 0.0 (only AMP reward).

    Attributes:
        trunk (nn.Sequential): MLP layers processing input features.
        amp_linear (nn.Linear): Final linear layer producing discriminator output.
        task_reward_lerp (float): Interpolation factor for combining rewards.
    """

    def __init__(self, input_dim, amp_reward_coef, hidden_layer_sizes, device, task_reward_lerp=0.0):
        super().__init__()

        self.device = device
        self.input_dim = input_dim

        self.amp_reward_coef = amp_reward_coef
        amp_layers = []
        curr_in_dim = input_dim
        for hidden_dim in hidden_layer_sizes:
            amp_layers.append(nn.Linear(curr_in_dim, hidden_dim))
            amp_layers.append(nn.ReLU())
            curr_in_dim = hidden_dim
        self.trunk = nn.Sequential(*amp_layers).to(device)
        self.amp_linear = nn.Linear(hidden_layer_sizes[-1], 1).to(device)

        self.trunk.train()
        self.amp_linear.train()

        self.task_reward_lerp = task_reward_lerp

    def forward(self, x):
        """
        Forward pass through the discriminator network.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Discriminator output logits with shape (batch_size, 1).
        """
        h = self.trunk(x)
        d = self.amp_linear(h)
        return d

    def compute_grad_pen(self, expert_state, expert_next_state, lambda_=10):
        """
        Compute gradient penalty for the expert data, used to regularize the discriminator.

        Args:
            expert_state (torch.Tensor): Batch of expert states.
            expert_next_state (torch.Tensor): Batch of expert next states.
            lambda_ (float, optional): Gradient penalty coefficient. Defaults to 10.

        Returns:
            torch.Tensor: Scalar gradient penalty loss.
        """
        expert_data = torch.cat([expert_state, expert_next_state], dim=-1)
        expert_data.requires_grad = True

        disc = self.amp_linear(self.trunk(expert_data))
        ones = torch.ones(disc.size(), device=disc.device)
        grad = autograd.grad(
            outputs=disc, inputs=expert_data, grad_outputs=ones, create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        # Enforce that the grad norm approaches 0.
        grad_pen = lambda_ * (grad.norm(2, dim=1) - 0).pow(2).mean()
        return grad_pen

    def predict_amp_reward(self, state, next_state, task_reward, normalizer=None):
        """
        Predict the AMP reward given current and next states, optionally interpolated with a task reward.

        Args:
            state (torch.Tensor): Current state tensor.
            next_state (torch.Tensor): Next state tensor.
            task_reward (torch.Tensor): Task-specific reward tensor.
            normalizer (optional): Normalizer object to normalize input states before prediction.

        Returns:
            tuple:
                - reward (torch.Tensor): Predicted AMP reward (optionally interpolated) with shape (batch_size,).
                - d (torch.Tensor): Raw discriminator output logits with shape (batch_size, 1).
        """
        # AMP reward inference should run with deterministic layers disabled (eval mode).
        # Do NOT force `.train()` here; the caller (runner/algorithm) controls the mode.
        was_training = self.training
        with torch.no_grad():
            self.eval()
            if normalizer is not None:
                state = normalizer.normalize_torch(state, self.device)
                next_state = normalizer.normalize_torch(next_state, self.device)

            d = self.amp_linear(self.trunk(torch.cat([state, next_state], dim=-1)))
            reward = self.amp_reward_coef * torch.clamp(1 - (1 / 4) * torch.square(d - 1), min=0)
            if self.task_reward_lerp > 0:
                reward = self._lerp_reward(reward, task_reward.unsqueeze(-1))

        if was_training:
            self.train()
        return reward.squeeze(), d

    def _lerp_reward(self, disc_r, task_r):
        """
        Linearly interpolate between discriminator reward and task reward.

        Args:
            disc_r (torch.Tensor): Discriminator reward.
            task_r (torch.Tensor): Task reward.

        Returns:
            torch.Tensor: Interpolated reward.
        """
        r = (1.0 - self.task_reward_lerp) * disc_r + self.task_reward_lerp * task_r
        return r


def get_ankle_remove_indices(obs_dim: int):
    """根据 AMP 观测维度返回需要移除的 ankle 关节索引。

    - 52 维: 20DoF 旧格式
    - 64 维: 25DoF GMR 格式
    其他维度: 不移除任何索引
    """
    # 这里目前返回空集，相当于不做任何裁剪。
    # 如果后续你想像原 HumanMimic 一样去掉 ankle 相关维度，可以把 amp_ppo_WGAN_GP.py 里的实现拷过来。
    return torch.tensor([], dtype=torch.long, device="cpu")




class Discriminator_WGAN_GP(nn.Module):
    """

    """

    def __init__(self, input_dim, amp_reward_coef, hidden_layer_sizes, device, task_reward_lerp=0.0):
        super().__init__()

        self.device = device
        self.input_dim = input_dim

        self.amp_reward_coef = amp_reward_coef
        amp_layers = []
        curr_in_dim = input_dim
        for hidden_dim in hidden_layer_sizes:
          #  amp_layers.append(nn.Linear(curr_in_dim, hidden_dim))
           # amp_layers.append(nn.ReLU())
            amp_layers.append(nn.Linear(curr_in_dim, hidden_dim))
            amp_layers.append(nn.ELU())
            curr_in_dim = hidden_dim
        self.trunk = nn.Sequential(*amp_layers).to(device)
        # Pairwise version: output layer without bias (as in Pairwise-Motion-Prior)
        self.amp_linear = nn.Linear(hidden_layer_sizes[-1], 1, bias=False).to(device)

        self.trunk.train()
        self.amp_linear.train()

        self.task_reward_lerp = task_reward_lerp
        
        # Add reward normalizer for pairwise version (as in Pairwise-Motion-Prior)
        # from rsl_rl.utils import utils
        # self.reward_norm = utils.Normalizer(1)

    def forward(self, x):
        """
        Forward pass through the discriminator network.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Discriminator output logits with shape (batch_size, 1).
        """
        h = self.trunk(x)
        d = self.amp_linear(h)
        return d

# --- 新增方法: HumanMimic Soft-Boundary Loss ---
   
    def compute_humanmimic_loss(self, d_real, d_fake, eta=0.2):
        """
        计算 HumanMimic 的 Soft-Boundary Wasserstein Loss。
        L = -E[tanh(eta * D_real)] + E[tanh(eta * D_fake)]
        
        Args:
            d_real: 真实数据(Expert)的 Discriminator 输出
            d_fake: 生成数据(Policy)的 Discriminator 输出
            eta: 软边界系数，论文推荐 0.1 ~ 0.5
        """
        loss_real = -torch.mean(torch.tanh(eta * d_real))
        loss_fake = torch.mean(torch.tanh(eta * d_fake))
        return loss_real + loss_fake



    def compute_grad_pen(self, expert_state, expert_next_state, policy_state, policy_next_state, lambda_=10):
        """
         01.16 修改版本 
        计算梯度惩罚。
        HumanMimic 使用 One-sided Gradient Penalty (单边惩罚)。

        """
        expert_data = torch.cat([expert_state, expert_next_state], dim=-1)
        policy_data = torch.cat([policy_state, policy_next_state], dim=-1)
        
        # 1. 插值 (Interpolation)
        alpha = torch.rand(expert_data.size(0), 1, device=expert_data.device)
        interpolates = alpha * expert_data + (1 - alpha) * policy_data
        interpolates.requires_grad = True

        # 2. 前向传播
        disc_interpolates = self.forward(interpolates)

        # 3. 计算梯度
        ones = torch.ones(disc_interpolates.size(), device=disc_interpolates.device)
        gradients = autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

# --- HumanMimic 修改 2: One-sided Gradient Penalty ---
        # 标准 WGAN-GP (Two-sided): (norm - 1)^2
        # HumanMimic (One-sided): max(0, norm - 1)^2
        # 允许梯度范数小于 1，只惩罚大于 1 的部分。
        
        grad_norms = gradients.norm(2, dim=1)
        # 使用 ReLU 来实现 max(0, x)
        grad_pen = lambda_ * torch.mean(torch.nn.functional.relu(grad_norms - 1) ** 2)

        return grad_pen



    def predict_amp_reward(self, state, next_state, task_reward, normalizer=None):
        """
         预测 AMP 奖励。
        HumanMimic 使用指数奖励 exp(d)。
        """
        # AMP reward inference should run with deterministic layers disabled (eval mode).
        # Do NOT force `.train()` here; the caller (runner/algorithm) controls the mode.
        was_training = self.training
        with torch.no_grad():
            self.eval()
            # 移除特定索引以匹配归一化器的维度（与Discriminator_baseline保持一致）
            # 根据观测维度自动选择正确的ankle索引 (52维20DOF or 64维25DOF)
            obs_dim = state.shape[1]
            remove_idx = get_ankle_remove_indices(obs_dim)
            keep_idx = torch.tensor([i for i in range(obs_dim) if i not in remove_idx.tolist()])
            state, next_state = state[:, keep_idx], next_state[:, keep_idx]
            
            if normalizer is not None:
                state = normalizer.normalize_torch(state, self.device)
                next_state = normalizer.normalize_torch(next_state, self.device)

            #d = self.amp_linear(self.trunk(torch.cat([state, next_state], dim=-1)))
            #reward = self.amp_reward_coef * torch.clamp(1 - (1 / 4) * torch.square(d - 1), min=0)

            d = self.amp_linear(self.trunk(torch.cat([state, next_state], dim=-1)))

            # --- HumanMimic 修改 3: 奖励计算 ---
            # 1. 禁用 reward_norm.update (WGAN 不需要归一化输出分布)
            # self.reward_norm.update(d.cpu().numpy()) 
            
            # 2. 使用 exp(d) 替代 sigmoid
            # 为了防止数值爆炸，建议加一个 clamp。
            # d 的值通常会被 Soft-Boundary Loss 限制在 [-1/eta, 1/eta] 之间，
            # 但 inference 时为了安全起见，我们截断一下最大值。
            
            # 原始 AMP: reward = self.amp_reward_coef * torch.clamp(1 - 0.25 * (d - 1)**2, min=0)
            # 你的代码: reward = self.amp_reward_coef * torch.sigmoid(d)
            
            # HumanMimic:
            reward = self.amp_reward_coef * torch.exp(torch.clamp(d, max=10.0))

            if self.task_reward_lerp > 0:
                reward = self._lerp_reward(reward, task_reward.unsqueeze(-1))

        if was_training:
            self.train()
        return reward.squeeze(), d

    def _lerp_reward(self, disc_r, task_r):
        """
        Linearly interpolate between discriminator reward and task reward.

        Args:
            disc_r (torch.Tensor): Discriminator reward.
            task_r (torch.Tensor): Task reward.

        Returns:
            torch.Tensor: Interpolated reward.
        """
        r = (1.0 - self.task_reward_lerp) * disc_r + self.task_reward_lerp * task_r
        return r

