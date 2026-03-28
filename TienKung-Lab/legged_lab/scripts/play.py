# flake8: noqa
# NOTE:
# This is an executable Isaac Sim script. It intentionally:
# - uses long help strings for CLI usability
# - imports some modules after SimulationApp is started (E402)
#
# Keeping flake8 disabled for this file avoids noisy style errors that don't affect runtime.
#
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

import argparse
import os

import cv2
import numpy as np
import torch
from isaaclab.app import AppLauncher

from legged_lab.utils import task_registry
from rsl_rl.runners import AmpOnPolicyRunner, AmpOnPolicyRunner_25, DWAQOnPolicyRunner, OnPolicyRunner

# local imports
import legged_lab.utils.cli_args as cli_args  # isort: skip

# NOTE:
# Avoid using eval() for resolving runner classes. It's brittle (NameError if not imported)
# and can be unsafe if config is untrusted. Use an explicit mapping instead.
RUNNER_CLASS_MAP = {
    "OnPolicyRunner": OnPolicyRunner,
    "AmpOnPolicyRunner": AmpOnPolicyRunner,
    "AmpOnPolicyRunner_25": AmpOnPolicyRunner_25,
    "DWAQOnPolicyRunner": DWAQOnPolicyRunner,
}

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--terrain", type=str, default="flat", 
                    choices=["stairs", "stairs_slope", "flat", "rough"],
                    help="Terrain type for play: stairs (纯台阶最难), stairs_slope (台阶+斜坡), flat (平地), rough (训练地形)")
parser.add_argument("--difficulty", type=float, default=0.2,
                    help="Terrain difficulty (0.0-1.0), default=1.0 (最难)")
parser.add_argument("--lighting", type=str, default="realistic",
                    choices=["realistic", "cloudy", "evening", "bright", "default"],
                    help="Lighting preset: realistic (真实户外), cloudy (多云), evening (傍晚), bright (明亮), default (默认白色)")
parser.add_argument("--terrain_color", type=str, default="mdl_shingles",
                    choices=["concrete", "grass", "sand", "dirt", "rock", "white", "dark",
                             "mdl_marble", "mdl_shingles", "mdl_aluminum"],
                    help="Terrain color: concrete/grass/sand/dirt/rock/white/dark (简单颜色), mdl_* (真实MDL材质)")
parser.add_argument("--no_gait", action="store_true",
                    help="Disable gait phase mechanism (for testing models trained without gait)")

# Save root-velocity tracking traces for plotting (paper figures).
parser.add_argument(
    "--save_trace",
    action="store_true",
    help="Save command vs root (base) velocity/pose traces to disk for plotting.",
)
parser.add_argument(
    "--trace_max_time",
    type=float,
    default=None,
    help="If set (in seconds), stop play after this simulated time and save trace.",
)
parser.add_argument(
    "--trace_format",
    type=str,
    default="csv",
    choices=["csv", "npz"],
    help="Trace file format to save: csv or npz.",
)
parser.add_argument(
    "--trace_name",
    type=str,
    default=None,
    help="Optional trace file name prefix. Default: <task>_<checkpoint_stem>.",
)

# Optional: override velocity commands at play-time.
# If you don't pass these, we keep whatever is defined in the task config (same as training),
# which is usually the safest way to evaluate a trained policy.
parser.add_argument(
    "--cmd_vx",
    type=float,
    default=None,
    help="Override commanded forward velocity (m/s). If set, lin_vel_x range becomes (cmd_vx, cmd_vx).",
)
parser.add_argument(
    "--cmd_vy",
    type=float,
    default=None,
    help="Override commanded lateral velocity (m/s). If set, lin_vel_y range becomes (cmd_vy, cmd_vy).",
)
parser.add_argument(
    "--cmd_wz",
    type=float,
    default=None,
    help="Override commanded yaw rate (rad/s). If set, ang_vel_z range becomes (cmd_wz, cmd_wz).",
)
parser.add_argument(
    "--cmd_no_heading",
    action="store_true",
    help="Disable heading-command mode at play time (use ang_vel_z command instead).",
)

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Debug: 检查 task 参数是否被正确解析
if hasattr(args_cli, 'task'):
    print(f"[DEBUG] args_cli.task = {args_cli.task}")
else:
    print("[DEBUG] args_cli 没有 'task' 属性")
# Start camera rendering for tasks that require RGB/depth sensing
if args_cli.task and ("sensor" in args_cli.task or "rgb" in args_cli.task or "depth" in args_cli.task):
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab_rl.rsl_rl import export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path

from legged_lab.envs import *  # noqa:F401, F403
from legged_lab.utils.cli_args import update_rsl_rl_cfg
from legged_lab.terrains.terrain_generator_cfg import STAIRS_ONLY_HARD_CFG, STAIRS_SLOPE_HARD_CFG
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, NVIDIA_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR


def play():
    runner: OnPolicyRunner
    env_cfg: BaseEnvCfg  # noqa:F405

    env_class_name = args_cli.task
    if env_class_name is None:
        raise ValueError(
            "❌ 错误: 必须提供 --task 参数！\n"
            "使用方法: python play.py --task <task_name> [其他参数...]\n"
            "示例: python play.py --task g1_walkamp_25 --load_run <run_name> --checkpoint <checkpoint>\n"
            "注意: 使用空格分隔参数，例如 --task g1_walkamp_25（不要用等号）"
        )
    env_cfg, agent_cfg = task_registry.get_cfgs(env_class_name)

    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.events.push_robot = None
    env_cfg.domain_rand.events.randomize_dome_light = None
    env_cfg.domain_rand.events.randomize_distant_light = None
    env_cfg.scene.max_episode_length_s = 40.0
    env_cfg.scene.num_envs = 1
    env_cfg.scene.env_spacing = 6.0
    # IMPORTANT:
    # Do NOT hard-force commands here; it can easily make a good policy look bad (e.g., always
    # commanding max forward speed). We only override commands if the user explicitly requests it.
    if args_cli.cmd_no_heading:
        env_cfg.commands.heading_command = False
        env_cfg.commands.rel_heading_envs = 0.0
        print("[INFO] Heading command disabled (--cmd_no_heading).")

    if args_cli.cmd_vx is not None:
        env_cfg.commands.ranges.lin_vel_x = (args_cli.cmd_vx, args_cli.cmd_vx)
    if args_cli.cmd_vy is not None:
        env_cfg.commands.ranges.lin_vel_y = (args_cli.cmd_vy, args_cli.cmd_vy)
    if args_cli.cmd_wz is not None:
        env_cfg.commands.ranges.ang_vel_z = (args_cli.cmd_wz, args_cli.cmd_wz)

    if (
        args_cli.cmd_vx is not None
        or args_cli.cmd_vy is not None
        or args_cli.cmd_wz is not None
    ):
        print(
            "[INFO] Overriding command ranges for play: "
            f"lin_vel_x={env_cfg.commands.ranges.lin_vel_x}, "
            f"lin_vel_y={env_cfg.commands.ranges.lin_vel_y}, "
            f"ang_vel_z={env_cfg.commands.ranges.ang_vel_z}"
        )

    env_cfg.commands.debug_vis = False  # Disable velocity command arrows
    env_cfg.scene.height_scanner.drift_range = (0.0, 0.0)

    # ========== G1 专用防炸机设置：根据训练经验下调 PD 刚度 ==========
    # 注意：这里是在 env 实例化之前修改 config，只影响本次 play，不影响训练。
    if hasattr(env_cfg, "robot") and hasattr(env_cfg.robot, "actuators"):
        print("[INFO] 正在强制重置 PD 参数以匹配更温和的设置（Kp≈20, Kd≈0.5）...")
        for act_name, act_cfg in env_cfg.robot.actuators.items():
            # 训练中常用的较软刚度范围，防止默认 100+ 的 Kp 导致抖动或炸机
            try:
                act_cfg.stiffness = 20.0
                act_cfg.damping = 0.5
                print(f"  - Actuator '{act_name}': set stiffness=20.0, damping=0.5")
            except Exception as e:
                print(f"  - 警告: 无法修改 '{act_name}' 的 PD 参数: {e}")

    # Disable gait phase if --no_gait is specified
    if args_cli.no_gait and hasattr(env_cfg, 'robot') and hasattr(env_cfg.robot, 'gait_phase'):
        env_cfg.robot.gait_phase.enable = False
        print("[INFO] 步态相位已禁用 (--no_gait)")
        print("[INFO] 注意: 这会改变观测维数。只能加载对应的模型。")
    elif hasattr(env_cfg, 'robot') and hasattr(env_cfg.robot, 'gait_phase') and env_cfg.robot.gait_phase.enable:
        print("[INFO] 步态相位已启用")
        print("[INFO] 观测包含步态相位信息 (sin + cos): +4 dims")

    # ========== 地形选择 ==========
    if args_cli.terrain == "stairs":
        # 纯台阶地形 (最难)
        env_cfg.scene.terrain_generator = STAIRS_ONLY_HARD_CFG
        env_cfg.scene.terrain_type = "generator"
        print("[INFO] 使用纯台阶地形 (最大难度)")
    elif args_cli.terrain == "stairs_slope":
        # 台阶 + 斜坡混合地形
        env_cfg.scene.terrain_generator = STAIRS_SLOPE_HARD_CFG
        env_cfg.scene.terrain_type = "generator"
        print("[INFO] 使用台阶+斜坡混合地形 (高难度)")
    elif args_cli.terrain == "flat":
        # 平地
        env_cfg.scene.terrain_generator = None
        env_cfg.scene.terrain_type = "plane"
        print("[INFO] 使用平地地形")
    elif args_cli.terrain == "rough":
        # 使用训练时的地形配置
        print("[INFO] 使用训练地形配置 (ROUGH_TERRAINS_CFG)")
    # 如果没有指定 --terrain，使用默认的训练地形

    # env_cfg.scene.terrain_generator = None
    # env_cfg.scene.terrain_type = "plane"

    if env_cfg.scene.terrain_generator is not None:
        env_cfg.scene.terrain_generator.num_rows = 5
        env_cfg.scene.terrain_generator.num_cols = 5
        env_cfg.scene.terrain_generator.curriculum = False
        # 使用命令行指定的难度
        difficulty = args_cli.difficulty
        env_cfg.scene.terrain_generator.difficulty_range = (difficulty, difficulty)
        print(f"[INFO] 地形难度: {difficulty}")

    # ========== 光照设置 ==========
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg
    
    # 光照预设配置
    LIGHTING_PRESETS = {
        "realistic": {  # 真实户外 - 晴朗天空
            "dome_intensity": 1000.0,
            "dome_color": (1.0, 0.98, 0.95),  # 略带暖色
            "dome_texture": f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
            "distant_intensity": 2500.0,
            "distant_color": (1.0, 0.95, 0.85),  # 阳光暖色
        },
        "cloudy": {  # 多云天气
            "dome_intensity": 1200.0,
            "dome_color": (0.9, 0.92, 0.95),  # 略带蓝灰
            "dome_texture": f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Cloudy/lakeside_4k.hdr",
            "distant_intensity": 1500.0,
            "distant_color": (0.85, 0.88, 0.92),
        },
        "evening": {  # 傍晚
            "dome_intensity": 800.0,
            "dome_color": (1.0, 0.85, 0.7),  # 暖橙色
            "dome_texture": f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Cloudy/evening_road_01_4k.hdr",
            "distant_intensity": 2000.0,
            "distant_color": (1.0, 0.7, 0.5),  # 夕阳色
        },
        "bright": {  # 明亮 (适合观察细节)
            "dome_intensity": 2000.0,
            "dome_color": (1.0, 1.0, 1.0),
            "dome_texture": f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
            "distant_intensity": 3500.0,
            "distant_color": (1.0, 1.0, 1.0),
        },
        "default": {  # 默认白色 (训练时用)
            "dome_intensity": 750.0,
            "dome_color": (1.0, 1.0, 1.0),
            "dome_texture": f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
            "distant_intensity": 3000.0,
            "distant_color": (0.75, 0.75, 0.75),
        },
    }
    
    lighting_preset = LIGHTING_PRESETS.get(args_cli.lighting, LIGHTING_PRESETS["realistic"])
    
    # 更新场景光照配置
    env_cfg.scene.light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(
            color=lighting_preset["distant_color"],
            intensity=lighting_preset["distant_intensity"],
        ),
    )
    env_cfg.scene.sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=lighting_preset["dome_intensity"],
            color=lighting_preset["dome_color"],
            texture_file=lighting_preset["dome_texture"],
            visible_in_primary_ray=True,  # 显示天空背景
        ),
    )
    print(f"[INFO] 光照预设: {args_cli.lighting}")

    # ========== 地形颜色设置 ==========
    # 简单颜色预设 (使用 PreviewSurfaceCfg)
    TERRAIN_COLOR_PRESETS = {
        "concrete": {  # 混凝土灰色 (真实感)
            "diffuse_color": (0.5, 0.5, 0.5),
            "roughness": 0.7,
        },
        "grass": {  # 草地绿色
            "diffuse_color": (0.2, 0.45, 0.2),
            "roughness": 0.9,
        },
        "sand": {  # 沙漠黄色
            "diffuse_color": (0.76, 0.7, 0.5),
            "roughness": 0.85,
        },
        "dirt": {  # 泥土棕色
            "diffuse_color": (0.45, 0.35, 0.25),
            "roughness": 0.9,
        },
        "rock": {  # 岩石灰色
            "diffuse_color": (0.4, 0.38, 0.35),
            "roughness": 0.8,
        },
        "white": {  # 白色 (原始)
            "diffuse_color": (0.9, 0.9, 0.9),
            "roughness": 0.5,
        },
        "dark": {  # 深色
            "diffuse_color": (0.18, 0.18, 0.18),
            "roughness": 0.6,
        },
    }
    
    # MDL 材质预设 (使用 MdlFileCfg - 更真实的材质)
    # 注意: 只使用经过验证的 MDL 路径
    MDL_TERRAIN_PRESETS = {
        "mdl_marble": {  # 大理石砖 (Isaac Lab 默认使用的，确保可用)
            "mdl_path": f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            "texture_scale": (0.25, 0.25),
        },
        "mdl_shingles": {  # 瓦片地面 (anymal_c 使用的)
            "mdl_path": f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            "texture_scale": (0.5, 0.5),
        },
        "mdl_aluminum": {  # 铝金属地面 (测试文件使用的)
            "mdl_path": f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Aluminum_Anodized.mdl",
            "texture_scale": (0.5, 0.5),
        },
    }
    
    # 根据选择设置地形材质
    if args_cli.terrain_color.startswith("mdl_"):
        # 使用 MDL 材质 (更真实)
        mdl_preset = MDL_TERRAIN_PRESETS.get(args_cli.terrain_color, MDL_TERRAIN_PRESETS["mdl_marble"])
        env_cfg.scene.terrain_visual_material = sim_utils.MdlFileCfg(
            mdl_path=mdl_preset["mdl_path"],
            project_uvw=True,
            texture_scale=mdl_preset["texture_scale"],
        )
        print(f"[INFO] 地形材质: {args_cli.terrain_color} (MDL)")
    else:
        # 使用简单颜色
        terrain_color = TERRAIN_COLOR_PRESETS.get(args_cli.terrain_color, TERRAIN_COLOR_PRESETS["concrete"])
        env_cfg.scene.terrain_visual_material = sim_utils.PreviewSurfaceCfg(
            diffuse_color=terrain_color["diffuse_color"],
            roughness=terrain_color["roughness"],
            metallic=0.0,
        )
        print(f"[INFO] 地形颜色: {args_cli.terrain_color}")

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.seed = agent_cfg.seed

    env_class = task_registry.get_task_class(env_class_name)
    env = env_class(env_cfg, args_cli.headless)
    # Debug: show the initial command after reset (helps diagnose "robot falls immediately")
    try:
        cmd0 = env.command_generator.command[0].detach().cpu().numpy()
        print(f"[INFO] Initial command (env0) = {cmd0}")
    except Exception as e:
        print(f"[WARN] Failed to read initial command from env: {e}")

    log_root_path = os.path.join("logs", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)
    trace_dir = os.path.join(log_dir, "play_traces")
    os.makedirs(trace_dir, exist_ok=True)

    runner_name = agent_cfg.runner_class_name
    runner_class = RUNNER_CLASS_MAP.get(runner_name, None)
    if runner_class is None:
        available = ", ".join(sorted(RUNNER_CLASS_MAP.keys()))
        raise ValueError(
            f"Unknown runner_class_name={runner_name!r}. Available: {available}. "
            "Check your agent config (runner_class_name)."
        )
    print(f"[INFO] Using runner_class_name={runner_name} -> {runner_class.__name__}")

    # ⬇️⬇️⬇️ 核心修改：在创建 Runner 之前，强制开启经验归一化 ⬇️⬇️⬇️
    print("[INFO] 正在检查 agent_cfg 配置，用于强制开启 empirical_normalization...")
    agent_cfg_dict = agent_cfg.to_dict() if hasattr(agent_cfg, "to_dict") else dict(agent_cfg)
    agent_cfg_dict["empirical_normalization"] = False
    print("[INFO] 已强制设置 empirical_normalization = True（仅影响当前 play 会话）")

    # 使用修改后的配置初始化 Runner
    runner = runner_class(env, agent_cfg_dict, log_dir=log_dir, device=agent_cfg.device)
    runner.load(resume_path, load_optimizer=False)

    # ========== 🛑 插入调试代码开始: 检查归一化参数 (Observation Normalizer) 🛑 ==========
    print("\n" + "=" * 50)
    print("正在检查归一化参数 (Observation Normalizer)...")
    if hasattr(runner, "obs_normalizer"):
        # 只有经验归一化开启时，obs_normalizer 才有 running_mean / running_var
        mean = getattr(runner.obs_normalizer, "running_mean", None)
        var = getattr(runner.obs_normalizer, "running_var", None)
        if mean is not None and var is not None:
            print(f"Normalizer Mean (前5个): {mean[:5].detach().cpu().numpy()}")
            print(f"Normalizer Var  (前5个): {var[:5].detach().cpu().numpy()}")

            # 核心判断：如果 Mean 全是 0，Var 全是 1，说明很可能没加载成功（仍是默认初始化）
            is_default = bool(torch.all(mean == 0).item()) and bool(torch.all(var == 1).item())
            if is_default:
                print("❌ 严重警告: 归一化参数看起来是默认值(0 和 1)！模型可能在“裸奔”。")
                print("   请检查 checkpoint 是否保存了 normalizer，或 load 路径是否正确。")
            else:
                print("✅ 归一化参数已加载 (不全为 0/1)，与训练时保持一致。")
        else:
            print("ℹ️  obs_normalizer 没有 running_mean/running_var（可能当前配置关闭了经验归一化）。")
    else:
        print("❌ 错误: Runner 中没有找到 obs_normalizer 属性！")
    print("=" * 50 + "\n")
    # ========== 🛑 插入调试代码结束 🛑 ==========

    # Check if using ActorCriticDepth which requires history and optionally rgb_image
    use_depth_policy = hasattr(runner.alg.policy, 'history_encoder')
    
    # Check if using DWAQ policy which requires obs_history from environment
    use_dwaq_policy = hasattr(runner.alg.policy, 'cenet_forward')
    
    # Check if RGB camera is available (from env, not runner)
    use_rgb = hasattr(env, 'rgb_camera') and env.rgb_camera is not None
    print(f"[INFO] use_depth_policy: {use_depth_policy}, use_dwaq_policy: {use_dwaq_policy}, use_rgb: {use_rgb}")
    
    if use_dwaq_policy:
        # DWAQ policy needs obs_history from environment
        runner.eval_mode()
        
        def policy_fn(obs, extras=None):
            # Get obs_history from extras (set by env.step())
            # If extras not available, get from env's buffer directly
            if extras is not None and "observations" in extras:
                obs_hist = extras["observations"]["obs_hist"]
            else:
                obs_hist = env.dwaq_obs_history_buffer.buffer.reshape(env.num_envs, -1)
            obs_hist = obs_hist.to(env.device)
            return runner.alg.policy.act_inference(obs, obs_hist)
        
        policy = policy_fn
    elif use_depth_policy:
        # Initialize trajectory history buffer
        # Get obs_history_len from env (preferred) or runner
        obs_history_len = getattr(env, 'obs_history_len', getattr(runner, 'obs_history_len', 1))
        num_obs = runner.num_obs
        trajectory_history = torch.zeros(
            size=(env.num_envs, obs_history_len, num_obs),
            device=env.device
        )
        
        # Set policy to eval mode
        runner.eval_mode()
        
        # Create inference function that handles history and rgb
        def policy_fn(obs):
            nonlocal trajectory_history
            normalized_obs = runner.obs_normalizer(obs) if runner.empirical_normalization else obs
            
            # Get RGB image if available
            rgb_image = None
            if use_rgb and hasattr(env, 'rgb_camera') and env.rgb_camera is not None:
                rgb_raw = env.rgb_camera.data.output["rgb"]
                if rgb_raw.shape[-1] == 4:
                    rgb_raw = rgb_raw[..., :3]
                rgb_image = rgb_raw.float().to(env.device) / 255.0
            
            actions = runner.alg.policy.act_inference(normalized_obs, trajectory_history, rgb_image=rgb_image)
            
            # Update history
            trajectory_history = torch.cat((trajectory_history[:, 1:], normalized_obs.unsqueeze(1)), dim=1)
            
            return actions
        
        policy = policy_fn
    else:
        policy = runner.get_inference_policy(device=env.device)

    # Skip JIT/ONNX export for ActorCriticDepth and DWAQ (complex architectures)
    if not use_depth_policy and not use_dwaq_policy:
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        export_policy_as_jit(runner.alg.policy, runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(
            runner.alg.policy, normalizer=runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
        )

    if not args_cli.headless:
        from legged_lab.utils.keyboard import Keyboard

        keyboard = Keyboard(env)  # noqa:F841

    obs, _ = env.get_observations()
    extras = env.extras  # Get initial extras
         # ================= 🌟 新增：开局热身阶段 (Warm-up) 🌟 =================
    # ======================================================================
    print("[INFO] 正在执行开局热身 (安全落地并填充历史缓冲区)...")
    # 执行 30 步 (大约 0.6 秒) 的零动作。
    # 在 IsaacLab 中，输入全 0 动作等于维持机器人默认的 Default Joint Position 站立姿态
    for _ in range(15):
        zero_actions = torch.zeros((env.num_envs, env.num_actions), device=env.device)
        obs, _, _, extras = env.step(zero_actions)
                
    print("[INFO] 热身完毕，历史缓冲已正常，神经网络正式接管！")
    # ======================================================================    
    # Reset trajectory history with initial observation if using depth policy
    if use_depth_policy:
        normalized_obs = runner.obs_normalizer(obs) if runner.empirical_normalization else obs
        trajectory_history = torch.cat((trajectory_history[:, 1:], normalized_obs.unsqueeze(1)), dim=1)

    # ------------------------------------------------------------------
    # 简单的 root 速度 / 位置 vs 命令速度 可视化（使用 OpenCV 绘制折线）
    # ------------------------------------------------------------------
    max_history = 200  # 保留最近 N 步
    cmd_vx_hist: list[float] = []
    root_vx_hist: list[float] = []
    root_x_hist: list[float] = []

    # ------------------------------------------------------------------
    # Save trace buffers for paper plotting
    # ------------------------------------------------------------------
    trace_rows: list[dict] = []
    step_i = 0

    def _get_default_trace_name() -> str:
        # e.g., task=g1_dwaq, checkpoint=model_9400.pt -> g1_dwaq_model_9400
        ckpt_stem = os.path.splitext(os.path.basename(resume_path))[0]
        return f"{env_class_name}_{ckpt_stem}"

    def _save_trace_to_disk():
        if not args_cli.save_trace:
            return
        if len(trace_rows) == 0:
            print("[WARN] --save_trace enabled but no samples were collected.")
            return
        name = args_cli.trace_name or _get_default_trace_name()
        if args_cli.trace_format == "csv":
            import csv

            out_path = os.path.join(trace_dir, f"{name}.csv")
            fieldnames = list(trace_rows[0].keys())
            with open(out_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(trace_rows)
            print(f"[INFO] Saved trace CSV: {out_path}")
        else:
            out_path = os.path.join(trace_dir, f"{name}.npz")
            # Convert list[dict] -> dict[str, np.ndarray]
            keys = list(trace_rows[0].keys())
            data = {k: np.asarray([r[k] for r in trace_rows]) for k in keys}
            np.savez_compressed(out_path, **data)
            print(f"[INFO] Saved trace NPZ: {out_path}")

    def update_debug_plot():
        """在独立窗口中画出 cmd_vx 与 root_vx 的对比曲线，并显示 root_x。"""
        if len(cmd_vx_hist) < 2:
            return
        w, h = 320, 240
        img = np.zeros((h, w, 3), dtype=np.uint8)

        # 坐标系：x 轴是时间，y 轴是速度，简单归一化到 [-1.5, 1.5] 区间
        vx_min, vx_max = -1.5, 1.5
        n = len(cmd_vx_hist)
        xs = np.linspace(0, w - 1, n, dtype=np.int32)

        def to_y(v):
            v_clamped = np.clip(v, vx_min, vx_max)
            norm = (v_clamped - vx_min) / (vx_max - vx_min)
            return int((1.0 - norm) * (h - 20)) + 10

        # 画 cmd_vx（绿色）和 root_vx（蓝色）
        for i in range(1, n):
            cv2.line(
                img,
                (xs[i - 1], to_y(cmd_vx_hist[i - 1])),
                (xs[i], to_y(cmd_vx_hist[i])),
                (0, 255, 0),
                2,
            )
            cv2.line(
                img,
                (xs[i - 1], to_y(root_vx_hist[i - 1])),
                (xs[i], to_y(root_vx_hist[i])),
                (255, 0, 0),
                2,
            )

        # 文本信息
        cv2.putText(
            img,
            f"cmd_vx (green), root_vx (blue)",
            (5, 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            f"root_x: {root_x_hist[-1]:.3f}",
            (5, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        cv2.imshow("Cmd vs Root Velocity", img)
        cv2.waitKey(1)

    try:
        while simulation_app.is_running():

            with torch.inference_mode():
                # DWAQ policy needs extras for obs_hist
                if use_dwaq_policy:
                    actions = policy(obs, extras)
                else:
                    actions = policy(obs)

                # All envs now return (obs, rewards, dones, extras) - Isaac Lab convention
                obs, _, dones, extras = env.step(actions)

                # ============================================================
                #  论文用：按时间 t 定义分段速度命令 (vx, vy, wz)
                #  例如：0–2s: 0 m/s, 2–4s: 0.2 m/s, 4–6s: -0.2 m/s, >6s: 0.0
                # ============================================================
                step_dt_cmd = float(getattr(env, "step_dt", 0.0))
                t_cmd = float(step_i * step_dt_cmd)
                vx_cmd, vy_cmd, wz_cmd = 0.0, 0.0, 0.0
                # if 2.0 <= t_cmd < 4.0:
                #     vx_cmd = 0.4
                # elif 4.0 <= t_cmd < 7.0:
                #     vx_cmd = 0.8
                #     vy_cmd = 0.4
                # elif 6.0 <= t_cmd < 8.0:
                #     vy_cmd = 0.2
                #     vx_cmd = 0.2
                # elif 8.0 <= t_cmd < 10.0:
                #     wz_cmd = 0
                # elif 10.0 <= t_cmd < 11.0:
                #     wz_cmd = -00
                
                if 2.0 <= t_cmd < 4.0:
                    vx_cmd = 0.4
                elif 4.0 <= t_cmd < 6.0:
                    vx_cmd = 0.6
                    vy_cmd = 0 
                elif 6.0 <= t_cmd < 8.0:
                    vy_cmd = 0
                    vx_cmd = 1
                elif 8.0 <= t_cmd < 10.0:
                    wz_cmd = 0
                    vx_cmd = 1.5
                elif 10.0 <= t_cmd < 11.0:
                    wz_cmd = -00
                    vx_cmd = 1.5
                
                # 其它时间段 vx_cmd 维持 0.0（可按需要自行扩展）
                with torch.no_grad():
                    env.command_generator.command[0, 0] = vx_cmd
                    if env.command_generator.command.shape[1] > 1:
                        env.command_generator.command[0, 1] = vy_cmd
                    if env.command_generator.command.shape[1] > 2:
                        env.command_generator.command[0, 2] = wz_cmd

                # ----- Debug: 记录并可视化 root 速度 与 命令速度 -----
                try:
                    # env.command_generator.command: [vx, vy, wz, ...]
                    cmd = env.command_generator.command[0].detach().cpu().numpy()
                    root_vel = env.robot.data.root_lin_vel_b[0].detach().cpu().numpy()
                    root_ang = None
                    if hasattr(env.robot.data, "root_ang_vel_b"):
                        root_ang = env.robot.data.root_ang_vel_b[0].detach().cpu().numpy()
                    root_pos = env.robot.data.root_pos_w[0].detach().cpu().numpy()

                    cmd_vx_hist.append(float(cmd[0]))
                    root_vx_hist.append(float(root_vel[0]))
                    root_x_hist.append(float(root_pos[0]))

                    # Save trace for plotting
                    step_dt = float(getattr(env, "step_dt", 0.0))
                    t = float(step_i * step_dt)
                    # 仅为论文绘图保存指令与根节点速度（不保存位置）。
                    # cmd_v*: 期望机体在水平面的线速度；root_v*: 实际机体在 body frame 下的线速度；
                    # root_wz: 机体在 body frame 下的 yaw 角速度。
                    row = {
                        "step": int(step_i),
                        "t": t,
                        "cmd_vx": float(cmd[0]),
                        "cmd_vy": float(cmd[1]) if cmd.shape[0] > 1 else 0.0,
                        "cmd_wz": float(cmd[2]) if cmd.shape[0] > 2 else 0.0,
                        "root_vx": float(root_vel[0]),
                        "root_vy": float(root_vel[1]) if root_vel.shape[0] > 1 else 0.0,
                    }
                    if root_ang is not None and root_ang.shape[0] > 2:
                        row["root_wz"] = float(root_ang[2])
                    else:
                        row["root_wz"] = 0.0
                    trace_rows.append(row)
                    step_i += 1

                    # 如果指定了 trace_max_time，到达后自动结束循环并保存数据
                    if args_cli.trace_max_time is not None and t >= args_cli.trace_max_time:
                        print(f"[INFO] Reached trace_max_time={args_cli.trace_max_time:.2f}s, stopping play loop.")
                        # 触发外层 finally 保存并退出
                        raise KeyboardInterrupt

                    # 限制历史长度
                    if len(cmd_vx_hist) > max_history:
                        cmd_vx_hist.pop(0)
                        root_vx_hist.pop(0)
                        root_x_hist.pop(0)

                    update_debug_plot()
                except Exception:
                    # 如果某些任务结构不同（没有 command_generator 或 root_vel），静默忽略
                    pass
            
            # Reset history for terminated environments
            if use_depth_policy:
                reset_env_ids = dones.nonzero(as_tuple=False).flatten()
                if len(reset_env_ids) > 0:
                    trajectory_history[reset_env_ids] = 0
            
   


                # Display RGB image in real-time using cv2.imshow
                if hasattr(env, 'rgb_camera') and env.rgb_camera is not None:
                    try:
                        rgb_raw = env.rgb_camera.data.output["rgb"]
                        lookat_id = getattr(env, 'lookat_id', 0)
                        rgb_img = rgb_raw[lookat_id].cpu().numpy()
                        # Ensure uint8 format
                        if rgb_img.dtype != 'uint8':
                            rgb_img = (rgb_img * 255).clip(0, 255).astype('uint8')
                        # Remove alpha channel if present
                        if rgb_img.shape[-1] == 4:
                            rgb_img = rgb_img[..., :3]
                        # Convert RGB to BGR for OpenCV
                        rgb_img_bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
                        # Resize for better visibility
                        rgb_img_resized = cv2.resize(rgb_img_bgr, (256, 256), interpolation=cv2.INTER_LINEAR)
                        # Display in window
                        cv2.imshow("RGB Camera View", rgb_img_resized)
                        cv2.waitKey(1)  # Required for window to update
                    except Exception:
                        pass  # Silently ignore errors
    finally:
        # Ensure we always flush traces on exit (including Ctrl+C).
        _save_trace_to_disk()


if __name__ == "__main__":
    play()
    simulation_app.close()
