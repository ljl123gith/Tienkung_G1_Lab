from __future__ import annotations

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

import legged_lab.mdp as mdp
from legged_lab.envs.base.base_env_config import BaseAgentCfg
from legged_lab.envs.g1.g1_config import G1FlatEnvCfg, G1RewardCfg
from legged_lab.terrains.terrain_generator_cfg import FLAT_TERRAINS_CFG


@configclass
class GaitCfg:
    gait_air_ratio_l: float = 0.38
    gait_air_ratio_r: float = 0.38
    gait_phase_offset_l: float = 0.38
    gait_phase_offset_r: float = 0.88
    gait_cycle: float = 0.85


@configclass
class G1WalkAmpRewardCfg(G1RewardCfg):
    # Removing some rewards from G1RewardCfg that might conflict or be less relevant for AMP walk on flat
    # However, user said "combine g1_flat rewards ... with walk task rewards"
    # So we keep G1RewardCfg base and add/override.
    
    # Gait phase rewards from TienKung
    gait_feet_frc_perio = RewTerm(func=mdp.gait_feet_frc_perio, weight=1.0, params={"delta_t": 0.02})
    gait_feet_spd_perio = RewTerm(func=mdp.gait_feet_spd_perio, weight=1.0, params={"delta_t": 0.02})
    gait_feet_frc_support_perio = RewTerm(func=mdp.gait_feet_frc_support_perio, weight=0.6, params={"delta_t": 0.02})
    
    # Adjust weights for specific terms if needed, e.g.
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    
    # TienKung has these, G1RewardCfg might have similar or different.
    # We will trust G1RewardCfg for the basics and just add the gait ones.


@configclass
class G1WalkAmpEnvCfg(G1FlatEnvCfg):
    """G1 flat terrain AMP-walk environment config."""
    
    reward = G1WalkAmpRewardCfg()
    gait = GaitCfg()

    motion_visualization_files: list[str] = [
        "legged_lab/envs/g1/datasets/motion_visualization/motion_walk_B1.txt",
        "legged_lab/envs/g1/datasets/motion_visualization/motion_walk_B3.txt",
        "legged_lab/envs/g1/datasets/motion_visualization/motion_walk_B4.txt",
        "legged_lab/envs/g1/datasets/motion_visualization/motion_walk_B9.txt",
        "legged_lab/envs/g1/datasets/motion_visualization/motion_walk_B22.txt",
    ]
    motion_visualization_index: int = 0
    motion_frame_duration: float = 0.033
    root_height_offset: float = 0.0
    # The motion_visualization txt is typically exported in MuJoCo joint order.
    # Options: "mujoco" or "lab"
    motion_dof_order: str = "mujoco"

    def __post_init__(self):
        super().__post_init__()
        # Keep defaults from base configs (usually 4096), but do NOT hard-force here.
        # This allows overriding via CLI: `train.py --num_envs <N>`.
        self.scene.terrain_generator = FLAT_TERRAINS_CFG
        
        # Ensure domain randomization for flat terrain matches expectations
        # (Inherited from G1FlatEnvCfg)



@configclass
class G1WalkAmpAgentCfg(BaseAgentCfg):
    """AMP training config for G1 rough walking."""

    experiment_name: str = "g1_walkamp"
    wandb_project: str = "g1_walkamp"
    runner_class_name: str = "AmpOnPolicyRunner"
    max_iterations: int = 20000

    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=1.0,
        noise_std_type="scalar",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="AMPPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        normalize_advantage_per_mini_batch=False,
        symmetry_cfg=None,
        rnd_cfg=None,
    )

    # AMP parameters
    amp_reward_coef = 5
    amp_motion_files = [
        "legged_lab/envs/g1/datasets/motion_amp_expert/g1_walkamp_B1.txt",
        "legged_lab/envs/g1/datasets/motion_amp_expert/g1_walkamp_B3.txt",
        "legged_lab/envs/g1/datasets/motion_amp_expert/g1_walkamp_B4.txt",
        "legged_lab/envs/g1/datasets/motion_amp_expert/g1_walkamp_B9.txt",
        "legged_lab/envs/g1/datasets/motion_amp_expert/g1_walkamp_B22.txt",
    ]
    amp_motion_weights = {
        "g1_walkamp_B1": 1.0,
        "g1_walkamp_B3": 1.0,
        "g1_walkamp_B4": 1.0,
        "g1_walkamp_B9": 1.0,
        "g1_walkamp_B22": 1.0,
    }
    amp_num_preload_transitions = 200000
    amp_task_reward_lerp = 0.7
    amp_discr_hidden_dims = [1024, 512, 256]
    min_normalized_std = [0.05] * 20
