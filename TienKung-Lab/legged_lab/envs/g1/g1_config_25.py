# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# Original code is licensed under BSD-3-Clause.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
# Modifications are licensed under BSD-3-Clause.
#
# This file contains code derived from Isaac Lab Project (BSD-3-Clause license)
# with modifications by Legged Lab Project (BSD-3-Clause license).

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.utils import configclass
import math  # 为 (-math.pi, math.pi) 提供支持


import legged_lab.mdp as mdp
from legged_lab.assets.unitree import G1_CFG_25
from legged_lab.envs.base.base_env_config import (  # noqa:F401
    BaseAgentCfg,
    BaseEnvCfg,
    BaseSceneCfg,
    DomainRandCfg,
    HeightScannerCfg,
    PhysxCfg,
    RewardCfg,
    RobotCfg,
    SimCfg,
    NormalizationCfg,   # ← 新增
    ObsScalesCfg,       # ← 新增
    CommandsCfg,        # ← 新增
    CommandRangesCfg,   # ← 新增
)
from legged_lab.terrains import GRAVEL_TERRAINS_CFG, ROUGH_TERRAINS_CFG


@configclass
class G1RewardCfg(RewardCfg):
    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=1.0, params={"std": 0.5})
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=1.0, params={"std": 0.5})
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    energy = RewTerm(func=mdp.energy, weight=-1e-3)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-5.0, # -1 -> -5
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names="(?!.*ankle.*).*"), "threshold": 1.0},
    )
    fly = RewTerm(
        func=mdp.fly, 
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"), "threshold": 1.0},
    )
    body_orientation_l2 = RewTerm(
        func=mdp.body_orientation_l2, params={"asset_cfg": SceneEntityCfg("robot", body_names=".*torso.*")}, weight=-2.0
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.15,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"), "threshold": 0.4},
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll.*"),
        },
    )
    feet_force = RewTerm(
        func=mdp.body_force,
        weight=-3e-3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"),
            "threshold": 500,
            "max_reward": 400,
        },
    )
    feet_too_near = RewTerm(
        func=mdp.feet_too_near_humanoid,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*ankle_roll.*"]), "threshold": 0.2},
    )
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*ankle_roll.*"])},
    )
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1_always,  # Always penalize hip yaw/roll deviation to prevent splay-footed gait
        weight=-0.15,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_hip_yaw.*", ".*_hip_roll.*"]
            )
        },
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1_always,  # Always penalize arm deviation, not just when standing
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*waist.*", ".*_shoulder_roll.*", ".*_shoulder_yaw.*", ".*_shoulder_pitch.*", ".*_elbow.*", ".*_wrist_roll.*"]
            )
        },
    )
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1_always,
        weight=-0.02,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_pitch.*", ".*_knee.*", ".*_ankle.*"])},
    )

@configclass
class G1WalkGaitRewardCfg25(G1RewardCfg):
    """
    在 25DoF G1 基础奖励上，叠加步态相位相关的奖励项。
    设计完全参考 g1_walkamp_cfg.G1WalkAmpRewardCfg，只是名字改成 25 版。
    """

    # # 步态周期性奖励：鼓励双脚在一个 gait 周期内接触力呈周期性变化
    # gait_feet_frc_perio = RewTerm(
    #     func=mdp.gait_feet_frc_perio,
    #     weight=1.0,
    #     params={"delta_t": 0.02},
    # )

    # # 步态周期性奖励：鼓励双脚速度在步态周期内呈合理的摆动/支撑模式
    # gait_feet_spd_perio = RewTerm(
    #     func=mdp.gait_feet_spd_perio,
    #     weight=1.0,
    #     params={"delta_t": 0.02},
    # )

    # # 支撑相的接触力周期性：鼓励在支撑相有稳定的支撑力
    # gait_feet_frc_support_perio = RewTerm(
    #     func=mdp.gait_feet_frc_support_perio,
    #     weight=0.6,
    #     params={"delta_t": 0.02},
    # )

    # # 如有需要，可以像 g1_walkamp 一样微调已有项的权重，例如：
    # # flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)




@configclass
class G1GaitCfg25:
    """
    G1 25-DoF gait-phase configuration.

    参数直接参考 `g1_walkamp_cfg.GaitCfg`，用于在 `G1Env_25` 中生成步态相位参考。
    """

    # 左右腿各自的“空中相位”比例（0~1）
    gait_air_ratio_l: float = 0.38
    gait_air_ratio_r: float = 0.38
    # 左右腿的相位偏移（单位：归一化相位 0~1）
    gait_phase_offset_l: float = 0.38
    gait_phase_offset_r: float = 0.88
    # 一个完整步态周期的时长（秒）
    gait_cycle: float = 0.85


@configclass
class G1FlatEnvCfg_25(BaseEnvCfg):
    """
    Flat terrain configuration for the 25-DoF G1 model.

    这里增加了 `use_gait_obs` 和 `gait`，用于在 `G1Env_25` 中启用步态相位观测，
    让 `g1_flat_25` 也可以像 `g1_walkamp` 一样使用步态相位作为参考信号。
    """

    # 是否在环境中附加 gait-phase 观测；打开后 obs 维度会增加，需要重新训练策略。
    use_gait_obs: bool = True
    # 步态配置，参数直接参考 g1_walkamp_cfg.GaitCfg。
    gait = G1GaitCfg25() 

    reward = G1WalkGaitRewardCfg25()
    normalization: NormalizationCfg = NormalizationCfg(
            obs_scales=ObsScalesCfg(
                lin_vel=1.0,
                ang_vel=1.0,
                projected_gravity=1.0,
                commands=1.0,
                joint_pos=1.0,
                joint_vel=1.0,
                actions=1.0,
                height_scan=1.0,
            ),
            clip_observations=100.0,
            clip_actions=100.0,
            height_scan_offset=0.5,
        )
    commands: CommandsCfg = CommandsCfg(
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.2,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=CommandRangesCfg(
            lin_vel_x=(-0, 0.5), lin_vel_y=(-00, 0), ang_vel_z=(-0, 0), heading=(-math.pi, math.pi)
        ),
    )


    def __post_init__(self):
        super().__post_init__()
        self.scene.height_scanner.prim_body_name = "torso_link"
        self.scene.robot = G1_CFG_25
        self.scene.terrain_type = "plane"
        self.scene.terrain_generator = GRAVEL_TERRAINS_CFG
        self.robot.terminate_contacts_body_names = [".*torso.*"]
        self.robot.feet_body_names = [".*ankle_roll.*"]
        self.domain_rand.events.add_base_mass.params["asset_cfg"].body_names = [".*torso.*"]


@configclass
class G1FlatAgentCfg_25(BaseAgentCfg):
    experiment_name: str = "g1_flat_25"
    wandb_project: str = "g1_flat_25"
    resume = True
    load_run = "2026-02-28_11-41-40"
    load_checkpoint = "model_4700.pt"

    
    def __post_init__(self):
        super().__post_init__()
        # Use ActorCritic with History Encoder (no LSTM needed with history buffer)
        self.policy.class_name = "ActorCritic"
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]



@configclass
class G1RoughEnvCfg_25(G1FlatEnvCfg_25):
    """
    G1 rough terrain environment configuration for blind walking.
    
    This configuration enables asymmetric actor-critic training where:
    - Actor: No height_scan (can be deployed on real robot without terrain sensor)
    - Critic: Has height_scan as privileged information for training
    
    Similar to G1RgbEnvCfg but without RGB camera (pure blind walking).
    """

    def __post_init__(self):
        super().__post_init__()
        
        # Terrain configuration - same as g1_rgb for rough terrains
        self.scene.terrain_generator = ROUGH_TERRAINS_CFG
        
        # Height scanner configuration - asymmetric AC (key difference from G1FlatEnvCfg)
        self.scene.height_scanner.enable_height_scan = True
        self.scene.height_scanner.critic_only = True  # Actor is blind, Critic has terrain info
        
        # Privileged information for Critic (same as g1_rgb)
        self.scene.privileged_info.enable_feet_info = True  # feet_pos + feet_vel in body frame (12 dim)
        self.scene.privileged_info.enable_feet_contact_force = True  # contact force 3D (6 dim)
        self.scene.privileged_info.enable_root_height = True  # root height (1 dim)
        
        # History length - 10 frames for temporal context (same as g1_rgb)
        self.robot.actor_obs_history_length = 10
        self.robot.critic_obs_history_length = 10
        
        # Action delay for sim-to-real transfer
        # self.domain_rand.action_delay.enable = True
        
        # # Reward weights for rough terrain walking
        # self.reward.feet_air_time.weight = 0.25
        # self.reward.track_lin_vel_xy_exp.weight = 1.5
        # self.reward.track_ang_vel_z_exp.weight = 1.5
        # self.reward.lin_vel_z_l2.weight = -0.25


@configclass
class G1RoughAgentCfg_25(BaseAgentCfg):
    """
    G1 rough terrain agent configuration.
    
    Uses ActorCritic with History Encoder for asymmetric actor-critic training.
    Similar architecture to G1RgbAgentCfg but without visual observation processing.
    """
    experiment_name: str = "g1_rough_25"
    wandb_project: str = "g1_rough_25"
    resume = True
    load_run = "2026-02-26_12-39-55"
    load_checkpoint = "model_9800.pt"

    def __post_init__(self):
        super().__post_init__()
        # Use ActorCritic with History Encoder (no LSTM needed with history buffer)
        self.policy.class_name = "ActorCritic"
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]
