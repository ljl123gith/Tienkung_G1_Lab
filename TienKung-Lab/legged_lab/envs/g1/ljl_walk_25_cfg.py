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
# Copyright (c) 2025-2026, The G1-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# with additional modifications by the G1-Lab Project,
# and is distributed under the BSD-3-Clause license.

import math
from pathlib import Path

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (  # noqa:F401
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlRndCfg,
    RslRlSymmetryCfg,
)

import legged_lab.mdp as mdp
from legged_lab.assets.unitree import G1_CFG_25 # Import G1 25DOF configuration
from legged_lab.envs.base.base_config import (
    ActionDelayCfg,
    BaseSceneCfg,
    CommandRangesCfg,
    CommandsCfg,
    DomainRandCfg,
    EventCfg,
    HeightScannerCfg,
    NoiseCfg,
    NoiseScalesCfg,
    NormalizationCfg,
    ObsScalesCfg,
    PhysxCfg,
    RobotCfg,
    SimCfg,
)
from legged_lab.terrains import GRAVEL_TERRAINS_CFG, ROUGH_TERRAINS_CFG  # noqa:F401


@configclass
class GaitCfg:
    """Gait configuration parameters for the G1 25DOF robot.
    
    These parameters control the walking gait pattern including timing and phase relationships.
    The values are scaled based on the robot's physical properties compared to reference data.
    """
    # 参数类型	    修改方向	    物理原因
    # 空中相位比例↑	增加0.04	    低KP需要更长时间完成摆动相动作
    # 步态周期↑	    增加0.2秒	关节响应变慢（hip/knee KP从700→100）
    # 脚力权重↓	    降低30%	    刚度降低后力控制精度下降，需放宽要求
    # 脚速权重↑	    增加40%	    补偿低KP导致的跟踪延迟，需强化速度控制
    # 新增稳定性奖励	权重0.8	    抑制低刚度系统常见的振荡现象

    # Gait air ratio: the proportion of time a leg spends in swing phase
    # Value range typically 0.3-0.5
    gait_air_ratio_l: float = 0.38  # Left leg swing phase ratio
    gait_air_ratio_r: float = 0.38  # Right leg swing phase ratio
    
    # Gait phase offset: controls the relative timing between left and right legs
    # The difference (~0.5) indicates alternating leg movement
    gait_phase_offset_l: float = 0.38   # Left leg phase offset
    gait_phase_offset_r: float = 0.88   # Right leg phase offset (0.88 - 0.38 ≈ 0.5)
    
    # Gait cycle: the complete gait period in seconds
    # Shorter cycle = higher step frequency
    gait_cycle: float = 0.85  # seconds

    # Reference robot parameters (TienKung)
    tienkung_mass = 65  # kg
    tk_thighs_long, tk_thighs_mass = 40, 3.951912+6.670795
    tk_calves_long, tk_calves_mass = 40, 2.312685
    tk_ankle_to_foot_long, tk_ankle_to_foot_mass = 6, 0.111609+1.017381

    # G1 robot parameters
    g1_mass = 35  # kg
    g1_thighs_long, g1_thighs_mass = 32.5, 1.52+1.702
    g1_calves_long, g1_calves_mass = 30.0, 1.932
    g1_ankle_to_foot_long, g1_ankle_to_foot_mass = 5.5, 0.074+0.608

    # Scaling ratios for transferring motion from reference to G1
    mass_ratio = g1_mass / tienkung_mass  # 35/65 ≈ 0.538

    tienkung_leg_length = 0.86  # m
    g1_leg_length = 0.69  # m
    length_ratio = g1_leg_length / tienkung_leg_length  # 0.69/0.86 ≈ 0.802

    # Time scaling based on pendulum model (T ∝ √L)
    time_scale = math.sqrt(length_ratio)  # √0.802 ≈ 0.896

    # Velocity scaling: v ∝ L/T → scale_v = length_ratio / time_scale
    velocity_scale = length_ratio / time_scale  # 0.802 / 0.896 ≈ 0.895

    # Force scaling: F ∝ m·a ∝ m·L/T² → scale_f = mass_ratio × length_ratio / (time_scale**2)
    force_scale = mass_ratio * length_ratio / (time_scale ** 2)  # 0.538 × 0.802 / 0.803 ≈ 0.537

    # Gait reward weight scaling based on physical properties
    swing_force_importance_scale = mass_ratio  # ≈ 0.538
    gait_feet_frc_perio_weight = 1.0 * swing_force_importance_scale  # ≈ 0.538

    speed_weight_scale = 1.0 / math.sqrt(mass_ratio)  # ≈ 1.363
    gait_feet_spd_perio_weight = 1.0 * speed_weight_scale  # ≈ 1.363

    stability_importance_scale = 1.0 / math.sqrt(mass_ratio)  # ≈ 1.363
    gait_feet_frc_support_perio_weight = 0.6 * stability_importance_scale  # ≈ 0.818


@configclass
class LiteRewardCfg:
    """Reward function configuration for the G1 25DOF walking task.
    
    The reward structure consists of:
    1. Primary task rewards: velocity tracking and gait generation
    2. Regularization rewards: penalize undesired behaviors
    """
    # 主任务奖励由【指令跟踪】和【步态生成】两大部分组成
    #     track_lin_vel_xy_exp (weight=5.0): Track desired linear velocity in XY plane
    #     track_ang_vel_z_exp (weight=5.0): Track desired angular velocity around Z axis

    # Primary task: Velocity tracking with exponential reward shape
    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=10, params={"std": 0.5})
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=5, params={"std": 0.5})
    
    # Regularization: Penalize vertical motion
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)

    # Regularization: Penalize base angular velocity in XY plane (roll/pitch rotation)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)

    # Regularization: Penalize energy consumption
    #energy = RewTerm(func=mdp.energy, weight=-1e-3)
    energy = RewTerm(func=mdp.energy, weight=0)
    # Regularization: Penalize joint accelerations to encourage smooth motion
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)

    # Regularization: Penalize action rate to encourage smooth actions
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    
    # Regularization: Penalize undesired body contacts (knees, shoulders, elbows, pelvis)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor", body_names=[".*knee.*", ".*shoulder_roll.*", ".*elbow.*", "pelvis"]
            ),
            "threshold": 1.0,
        },
    )

    # Regularization: Penalize pelvis orientation deviation from upright
    body_orientation_l2 = RewTerm(
        func=mdp.body_orientation_l2, params={"asset_cfg": SceneEntityCfg("robot", body_names="pelvis")}, weight=-2.0
    )

    # Regularization: Penalize base tilt from horizontal
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)

    # Large penalty for episode termination (falling, collision, etc.)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-10.0)

    # Feet behavior penalties
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
        },
    )
    
    # Penalize excessive foot impact forces
    feet_force = RewTerm(
        func=mdp.body_force,
        weight=-3e-3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"),
            "threshold": 500,
            "max_reward": 400,
        },
    )
    
    # Penalize feet being too close together
    feet_too_near = RewTerm(
        func=mdp.feet_too_near_humanoid,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*ankle_roll.*"]), "threshold": 0.2},
    )
    
    # Penalize feet stumbling on obstacles
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*ankle_roll.*"])},
    )

    # Penalize joint positions near limits
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)

    
    # Joint deviation penalties for different body parts
    # Hip and shoulder pitch joints
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*hip_roll.*joint",
                    ".*hip_yaw.*joint",
                    ".*shoulder_pitch.*joint",
                    ".*elbow.*joint",
                ],
            )
        },
    )

    # Arm joints (shoulder roll/yaw)
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*shoulder_roll.*joint",
                    ".*shoulder_yaw.*joint",
                ]
            )
        },
    )

    # Leg joints (hip pitch, knee pitch)
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.01,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*hip_pitch.*joint",
                    ".*knee.*joint",
                ],
            )
        },
    )

    # Ankle joints
    joint_deviation_ankles = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.01,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*ankle_pitch.*joint",
                    ".*ankle_roll.*joint",
                ],
            )
        },
    )

    # Waist joint deviation (new for G1 25DOF)
    joint_deviation_waist = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "waist_yaw_joint",
                    "waist_roll_joint",
                    "waist_pitch_joint",
                ],
            )
        },
    )
    
    # Ankle-specific penalties
   # ankle_torque = RewTerm(func=mdp.ankle_torque, weight=-0.0005)
    ankle_torque = RewTerm(func=mdp.ankle_torque, weight=0)
    ankle_action = RewTerm(func=mdp.ankle_action, weight=-0.001)

    # Hip action penalties
    hip_roll_action = RewTerm(func=mdp.hip_roll_action, weight=-0.1)
    hip_yaw_action = RewTerm(func=mdp.hip_yaw_action, weight=-0.1)

    # Feet separation penalty
    feet_y_distance = RewTerm(func=mdp.feet_y_distance, weight=-2.0)


@configclass
class G125WalkFlatEnvCfg:
    """Environment configuration for G1 25DOF walking on flat terrain with curriculum.
    
    This configuration defines:
    - Scene setup (robot, terrain, sensors)
    - Robot behavior parameters
    - Reward structure
    - Command ranges
    - Domain randomization
    - Simulation parameters
    """
    # Path to AMP motion files for visualization
    # 使用GMR转换后的数据
    amp_motion_files_display = [
        str(
            Path(__file__).parent
            / "datasets_ljl"
            / "motion_visualization"
            / "walk_0206_11_50_630_vis.txt"
        )
    ]
    
    device: str = "cuda:0"
    
    # Scene configuration with G1 25DOF robot
    scene: BaseSceneCfg = BaseSceneCfg(
        max_episode_length_s=20.0,  # Maximum episode duration
        num_envs=1024,  # Number of parallel environments (减少以节省GPU内存)
        env_spacing=2.5,  # Spacing between environments
        robot=G1_CFG_25,  # Use G1 25DOF robot configuration
        # terrain_type="generator",
        # terrain_generator=GRAVEL_TERRAINS_CFG,  # Use gravel terrain for training
        terrain_type="plane",  # Flat ground training
        terrain_generator=None,
        max_init_terrain_level=5,  # Maximum initial terrain difficulty
        height_scanner=HeightScannerCfg(
            enable_height_scan=False,  # Disable height scanning for now
            prim_body_name="pelvis",
            resolution=0.1,
            size=(1.6, 1.0),
            debug_vis=False,
            drift_range=(0.0, 0.0),
        ),
    )
    
    # Robot-specific configuration
    robot: RobotCfg = RobotCfg(
        actor_obs_history_length=10,  # Number of observation history steps for actor
        critic_obs_history_length=10,  # Number of observation history steps for critic
        action_scale=0.25,  # Scale factor for actions
        terminate_contacts_body_names=[".*knee.*", ".*shoulder_roll.*", ".*elbow.*", "pelvis"],
        feet_body_names=[".*ankle_roll.*"],
    )
    
    reward = LiteRewardCfg()  # Reward function configuration
    gait = GaitCfg()  # Gait parameters
    
    # Observation normalization parameters
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
    
    # Command generation configuration
    commands: CommandsCfg = CommandsCfg(
        resampling_time_range=(10.0, 10.0),  # Time range for resampling commands
        rel_standing_envs=0.2,  # Proportion of environments with standing command
        rel_heading_envs=1.0,  # Proportion of environments with heading command
        heading_command=True,  # Enable heading commands
        heading_control_stiffness=0.5,  # Stiffness for heading control
        debug_vis=True,  # Enable debug visualization
        ranges=CommandRangesCfg(
            lin_vel_x=(-0.6, 1.0),  # Linear velocity X range [m/s]
            lin_vel_y=(-0.5, 0.5),  # Linear velocity Y range [m/s]
            ang_vel_z=(-1.57, 1.57),  # Angular velocity Z range [rad/s]
            heading=(-math.pi, math.pi)  # Heading range [rad]
        ),
    )
    
    # Observation noise configuration
    noise: NoiseCfg = NoiseCfg(
        add_noise=True,
        noise_scales=NoiseScalesCfg(
            lin_vel=0.2,
            ang_vel=0.2,
            projected_gravity=0.05,
            joint_pos=0.01,
            joint_vel=1.5,
            height_scan=0.1,
        ),
    )
    
    # Domain randomization configuration
    domain_rand: DomainRandCfg = DomainRandCfg(
        events=EventCfg(
            # Randomize physics material properties at startup
            physics_material=EventTerm(
                func=mdp.randomize_rigid_body_material,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                    "static_friction_range": (0.6, 1.0),
                    "dynamic_friction_range": (0.4, 0.8),
                    "restitution_range": (0.0, 0.005),
                    "num_buckets": 64,
                },
            ),
            add_base_mass=None, 
            # # Randomize base mass at startup
            # add_base_mass=EventTerm(
            #     func=mdp.randomize_rigid_body_mass,
            #     mode="startup",
            #     params={
            #         "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
            #         "mass_distribution_params": (-5.0, 5.0),
            #         "operation": "add",
            #     },
            # ),
            # Reset base pose with randomization
            reset_base=EventTerm(
                func=mdp.reset_root_state_uniform,
                mode="reset",
                params={
                    "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
                    "velocity_range": {
                        "x": (-0.5, 0.5),
                        "y": (-0.5, 0.5),
                        "z": (-0.5, 0.5),
                        "roll": (-0.5, 0.5),
                        "pitch": (-0.5, 0.5),
                        "yaw": (-0.5, 0.5),
                    },
                },
            ),
            # Reset joint positions with randomization
            reset_robot_joints=EventTerm(
                func=mdp.reset_joints_by_scale,
                mode="reset",
                params={
                    "position_range": (0.5, 1.5),
                    "velocity_range": (0.0, 0.0),
                },
            ),
            # Apply random pushes during episode
            # push_robot=EventTerm(
            #     func=mdp.push_by_setting_velocity,
            #     mode="interval",
            #     interval_range_s=(10.0, 15.0),
            #     params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}},
            # ),
                        # Apply random pushes during episode
            # push_robot=EventTerm(
            #     func=mdp.push_by_setting_velocity,
            #     mode="interval",
            #     interval_range_s=(10.0, 15.0),
            #     params={
            #         "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
            #         "velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)},
            #     },
            # ),

        ),
        # action_delay=ActionDelayCfg(enable=False, params={"max_delay": 5, "min_delay": 0}),
    )
    
    # Simulation configuration
    sim: SimCfg = SimCfg(
        dt=0.005,  # Physics timestep [s]
        decimation=4,  # Number of physics steps per control step
        physx=PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15)
    )


@configclass
class G125WalkAgentCfg(RslRlOnPolicyRunnerCfg):
    """Training configuration for the PPO agent with AMP for G1 25DOF walking.
    
    This configuration defines:
    - Neural network architecture
    - PPO algorithm hyperparameters
    - AMP (Adversarial Motion Priors) settings
    - Training schedule and logging
    """
    seed = 42
    device = "cuda:0"
    num_steps_per_env = 24  # Number of steps to collect per environment before update
    max_iterations = 50000  # Maximum training iterations
    empirical_normalization = False  # Disable empirical observation normalization
    
    # Actor-Critic policy network configuration
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=1.0,  # Initial exploration noise
        noise_std_type="scalar",  # Scalar noise for all actions
        actor_hidden_dims=[512, 256, 128],  # Actor network layer sizes
        critic_hidden_dims=[512, 256, 128],  # Critic network layer sizes
        activation="elu",  # Activation function
    )
    
    # PPO algorithm configuration with AMP
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="AMPPPO",  # AMP-PPO with Wasserstein GAN gradient penalty
        value_loss_coef=1.0,  # Coefficient for value loss
        use_clipped_value_loss=True,  # Use clipped value loss
        clip_param=0.2,  # PPO clipping parameter
        entropy_coef=0.003,  # Entropy bonus coefficient for exploration
        num_learning_epochs=5,  # Number of epochs per update
        num_mini_batches=4,  # Number of mini-batches per epoch
        learning_rate=1.0e-3,  # Learning rate
        schedule="adaptive",  # Adaptive learning rate schedule
        gamma=0.99,  # Discount factor
        lam=0.95,  # GAE lambda parameter
        desired_kl=0.01,  # Target KL divergence for adaptive schedule
        max_grad_norm=1.0,  # Gradient clipping norm
        normalize_advantage_per_mini_batch=False,
        rnd_cfg=None,  # Random Network Distillation (disabled)
    )
    
    clip_actions = None
    save_interval = 1000  # Save model every N iterations
    runner_class_name = "AmpOnPolicyRunner"  # Use AMP runner
    experiment_name = "walk_g1"  # Experiment name for logging
    run_name = ""  # Optional run name
    logger = "tensorboard"  # Use TensorBoard for logging
    neptune_project = "walk"
    wandb_project = "walk"
    resume = False  # Resume from checkpoint
    load_run = ".*"  # Pattern for loading run
    load_checkpoint = "model_.*.pt"  # Pattern for loading checkpoint (most recent)

    # AMP (Adversarial Motion Priors) parameters
    amp_reward_coef = 3   #1（1.31） 2         # Coefficient for AMP reward (style reward weight)
    amp_motion_files = [
        str(
            Path(__file__).parent
            / "datasets_ljl"
            / "motion_visualization"
            / "walk_0206_11_50_630_vis.txt"  # 使用25DOF的GMR转换数据（64维）
        ) #G1DWAQ_Lab/TienKung-Lab/legged_lab/envs/g1/datasets_ljl/motion_visualization/walk_0131_100_1000.txt
    ]
    amp_num_preload_transitions = 200000  # Number of expert transitions to preload (减少内存占用)
    amp_task_reward_lerp = 0.7  # Interpolation factor between task and AMP rewards
    amp_discr_hidden_dims = [1024, 512, 256]  # AMP discriminator network layer sizes
    # Minimum standard deviation for each action (25 DOF)
    min_normalized_std = [0.05] * 25
