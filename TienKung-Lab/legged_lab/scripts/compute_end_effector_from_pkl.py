"""
从 pkl 文件计算末端位置（脚部和手部在 root 坐标系下的位置）
使用 Isaac Lab 环境来设置机器人状态并读取末端位置
"""
import pickle
import numpy as np
import torch
import argparse
from pathlib import Path
import sys

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from isaaclab.app import AppLauncher
from legged_lab.envs import task_registry
from legged_lab.utils.cli_args import update_rsl_rl_cfg
from scipy.spatial.transform import Rotation
from isaaclab.utils.math import quat_apply, quat_conjugate


def compute_end_effector_positions(input_pkl, output_pkl, task_name="g1_amp_25", headless=True):
    """从 pkl 文件计算并添加末端位置信息"""
    
    # 加载原始 pkl 数据
    print(f"📂 加载 pkl 文件: {input_pkl}")
    with open(input_pkl, "rb") as f:
        data = pickle.load(f)
    
    root_pos = data["root_pos"]
    root_rot = data["root_rot"]  # (N, 4) xyzw format
    dof_pos = data["dof_pos"]
    fps = data.get("fps", 30)
    
    num_frames = root_pos.shape[0]
    print(f"📊 数据帧数: {num_frames}")
    print(f"📊 关节数: {dof_pos.shape[1]}")
    
    # 检查是否已经有末端位置数据
    if "left_foot_pos_root" in data and data["left_foot_pos_root"] is not None:
        print("✅ pkl 文件已包含末端位置数据，跳过计算")
        return
    
    # 初始化 Isaac Lab 环境
    print("\n🔧 初始化 Isaac Lab 环境...")
    
    # 创建参数解析器（最小化配置）
    class Args:
        def __init__(self):
            self.task = task_name
            self.headless = headless
            self.num_envs = 1
            self.seed = 42
    
    args = Args()
    
    # 启动应用
    app_launcher = AppLauncher(headless=args.headless)
    simulation_app = app_launcher.app
    
    from legged_lab.envs import *
    
    # 获取环境配置
    env_cfg, agent_cfg = task_registry.get_cfgs(args.task)
    env_cfg.scene.num_envs = 1
    env_cfg.scene.env_spacing = 2.5
    env_cfg.scene.terrain_generator = None
    env_cfg.scene.terrain_type = "plane"
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.events.push_robot = None
    
    # 创建环境
    env_class = task_registry.get_task_class(args.task)
    env = env_class(env_cfg, args.headless)
    
    device = env.device
    
    # 获取末端 body IDs（从环境配置中）
    try:
        # 尝试从环境获取 body IDs
        if hasattr(env.robot, "data"):
            # 查找脚部和手部的 body IDs
            body_names = env.robot.root_physx_view.body_names
            
            # 常见的 body 名称（根据 G1 机器人模型）
            foot_body_names = ["left_foot", "right_foot", "l_foot", "r_foot", 
                              "left_ankle", "right_ankle", "l_ankle", "r_ankle"]
            hand_body_names = ["left_hand", "right_hand", "l_hand", "r_hand",
                              "left_wrist", "right_wrist", "l_wrist", "r_wrist"]
            
            left_foot_id = None
            right_foot_id = None
            left_hand_id = None
            right_hand_id = None
            
            for i, name in enumerate(body_names):
                name_lower = name.lower()
                if any(fn in name_lower for fn in ["left_foot", "l_foot", "left_ankle", "l_ankle"]):
                    if left_foot_id is None:
                        left_foot_id = i
                elif any(fn in name_lower for fn in ["right_foot", "r_foot", "right_ankle", "r_ankle"]):
                    if right_foot_id is None:
                        right_foot_id = i
                elif any(hn in name_lower for hn in ["left_hand", "l_hand", "left_wrist", "l_wrist"]):
                    if left_hand_id is None:
                        left_hand_id = i
                elif any(hn in name_lower for hn in ["right_hand", "r_hand", "right_wrist", "r_wrist"]):
                    if right_hand_id is None:
                        right_hand_id = i
            
            print(f"🔍 找到 body IDs:")
            print(f"   left_foot_id: {left_foot_id}")
            print(f"   right_foot_id: {right_foot_id}")
            print(f"   left_hand_id: {left_hand_id}")
            print(f"   right_hand_id: {right_hand_id}")
            
            if left_foot_id is None or right_foot_id is None:
                print("⚠️  无法自动找到脚部 body IDs，尝试使用环境中的配置...")
                # 尝试从环境配置中获取
                if hasattr(env, "feet_body_ids"):
                    left_foot_id = env.feet_body_ids[0] if len(env.feet_body_ids) > 0 else None
                    right_foot_id = env.feet_body_ids[1] if len(env.feet_body_ids) > 1 else None
                if hasattr(env, "ankle_body_ids"):
                    left_foot_id = env.ankle_body_ids[0] if len(env.ankle_body_ids) > 0 else left_foot_id
                    right_foot_id = env.ankle_body_ids[1] if len(env.ankle_body_ids) > 1 else right_foot_id
        else:
            raise AttributeError("无法访问 robot.data")
    except Exception as e:
        print(f"⚠️  无法自动获取 body IDs: {e}")
        print("   将尝试使用环境中的预定义 IDs...")
        left_foot_id = None
        right_foot_id = None
        left_hand_id = None
        right_hand_id = None
    
    # 初始化输出数组
    left_foot_pos_root = np.zeros((num_frames, 3), dtype=np.float64)
    right_foot_pos_root = np.zeros((num_frames, 3), dtype=np.float64)
    left_hand_pos_root = np.zeros((num_frames, 3), dtype=np.float64)
    right_hand_pos_root = np.zeros((num_frames, 3), dtype=np.float64)
    left_foot_height = np.zeros((num_frames,), dtype=np.float64)
    right_foot_height = np.zeros((num_frames,), dtype=np.float64)
    
    print(f"\n🔄 开始计算末端位置（共 {num_frames} 帧）...")
    
    # 处理每一帧
    for frame_idx in range(num_frames):
        if (frame_idx + 1) % 50 == 0:
            print(f"   处理进度: {frame_idx + 1}/{num_frames} ({100*(frame_idx+1)/num_frames:.1f}%)")
        
        # 设置机器人状态
        root_pos_frame = torch.tensor(root_pos[frame_idx], device=device, dtype=torch.float32)
        root_rot_frame = torch.tensor(root_rot[frame_idx], device=device, dtype=torch.float32)
        dof_pos_frame = torch.tensor(dof_pos[frame_idx], device=device, dtype=torch.float32)
        
        # 转换四元数格式：xyzw -> wxyz
        root_quat_wxyz = torch.tensor([
            root_rot_frame[3], root_rot_frame[0], 
            root_rot_frame[1], root_rot_frame[2]
        ], device=device, dtype=torch.float32)
        
        # 设置根状态
        root_state = torch.zeros((1, 13), device=device)
        root_state[0, 0:3] = root_pos_frame
        root_state[0, 3:7] = root_quat_wxyz
        env_ids = torch.tensor([0], device=device)
        env.robot.write_root_state_to_sim(root_state, env_ids)
        
        # 设置关节位置
        dof_pos_all = dof_pos_frame.unsqueeze(0)
        env.robot.write_joint_position_to_sim(dof_pos_all)
        
        # 步进仿真以更新状态
        env.sim.step()
        env.scene.update(dt=1.0/fps)
        
        # 读取 body 位置
        body_state_w = env.robot.data.body_state_w  # (num_envs, num_bodies, 13)
        root_state_w = env.robot.data.root_state_w  # (num_envs, 13)
        
        root_pos_w = root_state_w[0, 0:3]
        root_quat_w = root_state_w[0, 3:7]
        
        # 计算脚部位置（在 root 坐标系下）
        if left_foot_id is not None and left_foot_id < body_state_w.shape[1]:
            left_foot_pos_w = body_state_w[0, left_foot_id, 0:3]
            left_foot_pos_rel = left_foot_pos_w - root_pos_w
            left_foot_pos_root_frame = quat_apply(
                quat_conjugate(root_quat_w.unsqueeze(0)), 
                left_foot_pos_rel.unsqueeze(0)
            )[0]
            left_foot_pos_root[frame_idx] = left_foot_pos_root_frame.cpu().numpy()
            left_foot_height[frame_idx] = left_foot_pos_w[2].cpu().numpy()
        
        if right_foot_id is not None and right_foot_id < body_state_w.shape[1]:
            right_foot_pos_w = body_state_w[0, right_foot_id, 0:3]
            right_foot_pos_rel = right_foot_pos_w - root_pos_w
            right_foot_pos_root_frame = quat_apply(
                quat_conjugate(root_quat_w.unsqueeze(0)), 
                right_foot_pos_rel.unsqueeze(0)
            )[0]
            right_foot_pos_root[frame_idx] = right_foot_pos_root_frame.cpu().numpy()
            right_foot_height[frame_idx] = right_foot_pos_w[2].cpu().numpy()
        
        # 计算手部位置（在 root 坐标系下）
        if left_hand_id is not None and left_hand_id < body_state_w.shape[1]:
            left_hand_pos_w = body_state_w[0, left_hand_id, 0:3]
            left_hand_pos_rel = left_hand_pos_w - root_pos_w
            left_hand_pos_root_frame = quat_apply(
                quat_conjugate(root_quat_w.unsqueeze(0)), 
                left_hand_pos_rel.unsqueeze(0)
            )[0]
            left_hand_pos_root[frame_idx] = left_hand_pos_root_frame.cpu().numpy()
        
        if right_hand_id is not None and right_hand_id < body_state_w.shape[1]:
            right_hand_pos_w = body_state_w[0, right_hand_id, 0:3]
            right_hand_pos_rel = right_hand_pos_w - root_pos_w
            right_hand_pos_root_frame = quat_apply(
                quat_conjugate(root_quat_w.unsqueeze(0)), 
                right_hand_pos_rel.unsqueeze(0)
            )[0]
            right_hand_pos_root[frame_idx] = right_hand_pos_root_frame.cpu().numpy()
    
    print("✅ 计算完成！")
    
    # 保存结果到新的 pkl 文件
    print(f"\n💾 保存结果到: {output_pkl}")
    output_data = data.copy()
    output_data["left_foot_pos_root"] = left_foot_pos_root
    output_data["right_foot_pos_root"] = right_foot_pos_root
    output_data["left_hand_pos_root"] = left_hand_pos_root
    output_data["right_hand_pos_root"] = right_hand_pos_root
    output_data["left_foot_height"] = left_foot_height
    output_data["right_foot_height"] = right_foot_height
    
    with open(output_pkl, "wb") as f:
        pickle.dump(output_data, f)
    
    print("✅ 保存成功！")
    print(f"\n📊 输出数据包含的字段:")
    for key in output_data.keys():
        if output_data[key] is not None:
            if isinstance(output_data[key], np.ndarray):
                print(f"   {key}: shape={output_data[key].shape}")
            else:
                print(f"   {key}: {output_data[key]}")
    
    # 关闭应用
    simulation_app.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从 pkl 文件计算末端位置")
    parser.add_argument("--input_pkl", type=str, required=True, help="输入 pkl 文件路径")
    parser.add_argument("--output_pkl", type=str, required=True, help="输出 pkl 文件路径")
    parser.add_argument("--task", type=str, default="g1_amp_25", help="任务名称")
    parser.add_argument("--headless", action="store_true", help="无头模式")
    args = parser.parse_args()
    
    compute_end_effector_positions(
        args.input_pkl, 
        args.output_pkl, 
        args.task, 
        args.headless
    )

