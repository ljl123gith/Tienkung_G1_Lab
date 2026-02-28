import pickle
import json
import numpy as np
import argparse
from scipy.spatial.transform import Rotation

"""
使用方法：
python gmr_data_conversion_copy.py \
  --input_pkl /path/to/input.pkl \
  --output_txt /path/to/output.txt \
  --fps 30.0
参数说明：
--input_pkl: 输入的 pkl 文件路径
--output_txt: 输出的 txt 文件路径
--fps: 帧率

示例：
python gmr_data_conversion_copy.py \
  --input_pkl /abs/path/to/motion.pkl \
  --output_txt /abs/path/to/motion.txt \
  --fps 30.0
"""


def compute_ang_vel_wxyz(root_rot_wxyz: np.ndarray, dt: float) -> np.ndarray:
    """
    用 SciPy 计算角速度（避免依赖 IsaacLab/IsaacSim）。

    输入:
    - root_rot_wxyz: (N, 4) 四元数，格式为 w,x,y,z
    - dt: 相邻帧时间间隔

    输出:
    - ang_vel: (N-1, 3) 角速度（axis-angle/rotvec 除以 dt）
    """
    if root_rot_wxyz.ndim != 2 or root_rot_wxyz.shape[1] != 4:
        raise ValueError(
            f"root_rot_wxyz shape must be (N,4), got {root_rot_wxyz.shape}"
        )

    # SciPy Rotation 需要 (x,y,z,w)
    quat_xyzw = root_rot_wxyz[:, [1, 2, 3, 0]]
    rotations = Rotation.from_quat(quat_xyzw)
    # 相对旋转：r_rel = r_t^{-1} * r_{t+1}
    rel = rotations[:-1].inv() * rotations[1:]
    rotvec = rel.as_rotvec()  # (N-1, 3)
    return rotvec / dt


def detect_file_format(file_path):
    """检测文件格式：pickle 或 JSON"""
    try:
        with open(file_path, 'rb') as f:
            first_bytes = f.read(10)
            # JSON 文件通常以 { 或 [ 开头
            if first_bytes.startswith(b'{') or first_bytes.startswith(b'['):
                return 'json'
            # Pickle 文件有特定的魔数
            elif first_bytes.startswith(b'\x80'):
                return 'pickle'
            else:
                # 尝试作为文本文件读取
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read(100)
                    content_strip = content.strip()
                    if (
                        content_strip.startswith('{')
                        or content_strip.startswith('[')
                    ):
                        return 'json'
    except Exception:
        pass
    return 'pickle'  # 默认尝试 pickle


def load_motion_data(file_path):
    """加载运动数据，支持 pickle 和 JSON 格式"""
    file_format = detect_file_format(file_path)
    print(f"📁 检测到文件格式: {file_format}")

    if file_format == 'json':
        with open(file_path, 'r', encoding='utf-8') as f:
            json.load(f)
        print("⚠️  警告: 输入文件是 JSON 格式，不是原始的 pickle 文件。")
        print("    JSON 格式通常不包含末端位置等详细信息。")
        print("    ️请使用原始的 .pkl 文件作为输入。")
        return None, file_format

    # Pickle 格式
    with open(file_path, "rb") as f:
        motion_data = pickle.load(f)
    return motion_data, file_format


def convert_76d_to_64d(data_76d, joint_order='urdf'):
    """
    将76维数据转换为64维格式（兼容 play_amp_animation.py）

    Args:
        data_76d: (N, 76) 数组，76维格式的数据
        joint_order: 关节顺序，'urdf' 表示假设输入是URDF顺序

    Returns:
        data_64d: (N, 64) 数组，64维格式的数据
    """
    N = data_76d.shape[0]
    data_64d = np.zeros((N, 64), dtype=data_76d.dtype)

    # 基础字段（保持不变）
    data_64d[:, 0:3] = data_76d[:, 0:3]      # root_pos
    data_64d[:, 3:6] = data_76d[:, 3:6]      # euler_angles

    # 关节位置：假设输入是URDF顺序，需要重新排列
    # 注意：这里假设pkl中的dof_pos顺序与环境的joint顺序一致
    # 如果实际顺序不同，需要根据实际情况调整
    dof_pos_25 = data_76d[:, 6:31]  # 25个关节

    # 环境期望的顺序：
    # [0:6] left_leg, [6:12] right_leg, [12:15] waist,
    # [15:20] left_arm, [20:25] right_arm
    # 这里假设输入顺序就是：
    # left_leg(6) + right_leg(6) + waist(3) + left_arm(5) + right_arm(5)
    # 如果顺序不同，需要调整索引
    data_64d[:, 6:31] = dof_pos_25  # 直接复制（假设顺序一致）

    # 速度字段
    data_64d[:, 31:34] = data_76d[:, 31:34]  # root_lin_vel
    data_64d[:, 34:37] = data_76d[:, 34:37]  # root_ang_vel

    dof_vel_25 = data_76d[:, 37:62]  # 25个关节速度
    data_64d[:, 37:62] = dof_vel_25  # 直接复制（假设顺序一致）

    # 脚部高度
    data_64d[:, 62:63] = data_76d[:, 62:63]  # left_foot_height
    data_64d[:, 63:64] = data_76d[:, 63:64]  # right_foot_height

    # 注意：64维格式不包含末端位置（foot_pos_root, hand_pos_root）

    return data_64d


def convert_pkl_to_custom(input_pkl, output_txt, fps, output_format='full'):
    dt = 1.0 / fps

    # 加载数据
    motion_data, file_format = load_motion_data(input_pkl)
    if motion_data is None:
        print("❌ 无法处理 JSON 格式文件，请使用原始的 .pkl 文件")
        return

    # 打印所有可用的键
    print(f"\n📋 PKL 文件中的所有键: {list(motion_data.keys())}")

    # 基础字段（必须存在）
    root_pos = motion_data["root_pos"]
    root_rot = motion_data["root_rot"][:, [3, 0, 1, 2]]  # xyzw → wxyz
    dof_pos = motion_data["dof_pos"]

    # 计算速度
    root_lin_vel_w = (root_pos[1:] - root_pos[:-1]) / dt
    # 2. 将世界系线速度转换到基座局部系 (Local Base Frame)
    # root_rot 的格式是 w,x,y,z，而 scipy 需要 x,y,z,w
    rotations = Rotation.from_quat(root_rot[:-1, [1, 2, 3, 0]])
    root_lin_vel_b = rotations.inv().apply(root_lin_vel_w)

    # 替换原来的 root_lin_vel
    root_lin_vel = root_lin_vel_b

    # 角速度：用 SciPy 从相邻帧四元数得到相对旋转 rotvec，再除以 dt
    # 这样脚本就不依赖 isaaclab/isaacsim，方便在纯数据处理环境运行
    root_ang_vel = compute_ang_vel_wxyz(root_rot, dt)

    # Debug logs：帮助快速定位维度/类型问题
    print(
        f"ℹ️  root_pos shape={getattr(root_pos, 'shape', None)}, "
        f"root_rot shape={getattr(root_rot, 'shape', None)}, "
        f"dof_pos shape={getattr(dof_pos, 'shape', None)}"
    )
    print(
        f"ℹ️  root_lin_vel shape={getattr(root_lin_vel, 'shape', None)}, "
        f"root_ang_vel shape={getattr(root_ang_vel, 'shape', None)}, "
        f"dof_vel will be computed from dof_pos"
    )

    dof_vel = (dof_pos[1:] - dof_pos[:-1]) / dt

    # 转换为欧拉角
    euler_angles = Rotation.from_quat(
        root_rot[:-1, [1, 2, 3, 0]]
    ).as_euler('XYZ', degrees=False)
    euler_angles = np.unwrap(euler_angles, axis=0)

    # 提取所有可用字段（如果存在）
    # 尝试多种可能的字段名
    def get_field(data, possible_names, description):
        """尝试多个可能的字段名"""
        for name in possible_names:
            if name in data and data[name] is not None:
                value = data[name]
                if isinstance(value, np.ndarray):
                    print(f"✅ 找到 {description}: {name} (shape: {value.shape})")
                    return value
        print(f"⚠️  未找到 {description}，尝试的字段名: {possible_names}")
        return None

    # 脚部高度（世界坐标系z值）
    left_foot_height = get_field(
        motion_data,
        ["left_foot_height", "l_foot_height", "left_foot_z", "lfoot_height"],
        "left_foot_height"
    )
    right_foot_height = get_field(
        motion_data,
        ["right_foot_height", "r_foot_height", "right_foot_z", "rfoot_height"],
        "right_foot_height"
    )

    # 末端在root坐标系下的位置
    left_foot_pos_root = get_field(
        motion_data,
        [
            "left_foot_pos_root",
            "l_foot_pos_root",
            "left_foot_pos",
            "lfoot_pos_root",
        ],
        "left_foot_pos_root"
    )
    right_foot_pos_root = get_field(
        motion_data,
        [
            "right_foot_pos_root",
            "r_foot_pos_root",
            "right_foot_pos",
            "rfoot_pos_root",
        ],
        "right_foot_pos_root"
    )
    left_hand_pos_root = get_field(
        motion_data,
        [
            "left_hand_pos_root",
            "l_hand_pos_root",
            "left_hand_pos",
            "lhand_pos_root",
        ],
        "left_hand_pos_root"
    )
    right_hand_pos_root = get_field(
        motion_data,
        [
            "right_hand_pos_root",
            "r_hand_pos_root",
            "right_hand_pos",
            "rhand_pos_root",
        ],
        "right_hand_pos_root"
    )

    # 检查 local_body_pos（可能包含末端位置信息）
    local_body_pos = motion_data.get("local_body_pos", None)
    link_body_list = motion_data.get("link_body_list", None)
    if local_body_pos is not None and link_body_list is not None:
        print(f"ℹ️  找到 local_body_pos (shape: {local_body_pos.shape})")
        print(f"ℹ️  找到 link_body_list: {link_body_list}")
        print(
            "   提示: 如果 local_body_pos 包含末端位置，需要根据 link_body_list 提取"
        )

    # 构建输出数据列表
    output_parts = [
        root_pos[:-1],      # (N-1, 3) 根位置
        euler_angles,       # (N-1, 3) 根欧拉角
        dof_pos[:-1],       # (N-1, 25) 关节角度
        root_lin_vel,       # (N-1, 3) 根线速度
        root_ang_vel,       # (N-1, 3) 根角速度
        dof_vel,            # (N-1, 25) 关节速度
    ]

    # 添加脚部高度（如果存在）
    if left_foot_height is not None:
        if left_foot_height.ndim == 1:
            output_parts.append(
                left_foot_height[:-1].reshape(-1, 1)  # (N-1, 1)
            )
        else:
            output_parts.append(left_foot_height[:-1])
    else:
        print("   → 使用零填充")
        output_parts.append(np.zeros((root_pos.shape[0] - 1, 1)))

    if right_foot_height is not None:
        if right_foot_height.ndim == 1:
            output_parts.append(
                right_foot_height[:-1].reshape(-1, 1)  # (N-1, 1)
            )
        else:
            output_parts.append(right_foot_height[:-1])
    else:
        print("   → 使用零填充")
        output_parts.append(np.zeros((root_pos.shape[0] - 1, 1)))

    # 添加末端位置（如果存在）
    if left_foot_pos_root is not None:
        output_parts.append(left_foot_pos_root[:-1])  # (N-1, 3)
    else:
        print("   → 使用零填充")
        output_parts.append(np.zeros((root_pos.shape[0] - 1, 3)))

    if right_foot_pos_root is not None:
        output_parts.append(right_foot_pos_root[:-1])  # (N-1, 3)
    else:
        print("   → 使用零填充")
        output_parts.append(np.zeros((root_pos.shape[0] - 1, 3)))

    if left_hand_pos_root is not None:
        output_parts.append(left_hand_pos_root[:-1])  # (N-1, 3)
    else:
        print("   → 使用零填充")
        output_parts.append(np.zeros((root_pos.shape[0] - 1, 3)))

    if right_hand_pos_root is not None:
        output_parts.append(right_hand_pos_root[:-1])  # (N-1, 3)
    else:
        print("   → 使用零填充")
        output_parts.append(np.zeros((root_pos.shape[0] - 1, 3)))

    # 拼接所有数据
    data_output = np.concatenate(output_parts, axis=1)

    # 根据输出格式转换
    if output_format == 'visualization' or output_format == '64d':
        print("\n🔄 转换为64维格式（兼容 play_amp_animation.py）...")
        data_output = convert_76d_to_64d(data_output)
        print("✅ 转换完成")

    # 打印输出维度信息
    print(f"\n📊 输出数据维度: {data_output.shape}")
    print(f"   每帧包含: {data_output.shape[1]} 维")
    print(f"   总帧数: {data_output.shape[0]}")
    if output_format == 'visualization' or output_format == '64d':
        print("   ✅ 格式: 64维（兼容 play_amp_animation.py）")
    else:
        print("   ✅ 格式: 76维（完整数据，包含末端位置）")

    np.savetxt(output_txt, data_output, fmt='%f', delimiter=', ')
    with open(output_txt, 'r') as f:
        frames_data = f.readlines()

    frames_data_len = len(frames_data)
    with open(output_txt, 'w') as f:
        f.write('{\n')
        f.write('"LoopMode": "Wrap",\n')
        f.write(f'"FrameDuration": {1.0/fps:.3f},\n')
        f.write('"EnableCycleOffsetPosition": true,\n')
        f.write('"EnableCycleOffsetRotation": true,\n')
        f.write('"MotionWeight": 0.5,\n\n')
        f.write('"Frames":\n[\n')

        for i, line in enumerate(frames_data):
            line_start_str = '  ['
            if i == frames_data_len - 1:
                f.write(line_start_str + line.rstrip() + ']\n')
            else:
                f.write(line_start_str + line.rstrip() + '],\n')

        f.write(']\n}')
    print(f"✅ Successfully converted {input_pkl} to {output_txt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="将 pkl 文件转换为 txt 格式（JSON-like）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
输出格式选项:
  full (默认): 76维完整数据，包含所有字段（包括末端位置）
  visualization 或 64d: 64维格式，兼容 play_amp_animation.py

示例:
  # 输出完整76维数据
  python gmr_data_conversion_copy.py \\
    --input_pkl input.pkl \\
    --output_txt output.txt \\
    --fps 30.0

  # 输出64维可视化兼容格式
  python gmr_data_conversion_copy.py \\
    --input_pkl input.pkl \\
    --output_txt output.txt \\
    --fps 30.0 \\
    --format visualization
        """
    )
    parser.add_argument(
        "--input_pkl", type=str, required=True, help="输入 pkl 文件路径"
    )
    parser.add_argument(
        "--output_txt", type=str, required=True, help="输出 txt 文件路径"
    )
    parser.add_argument("--fps", type=float, default=30.0, help="帧率")
    parser.add_argument(
        "--format",
        type=str,
        default="full",
        choices=["full", "visualization", "64d"],
        help=(
            "输出格式: 'full' (76维) 或 'visualization'/'64d' "
            "(64维，兼容可视化)"
        ),
    )
    args = parser.parse_args()

    convert_pkl_to_custom(
        args.input_pkl,
        args.output_txt,
        args.fps,
        args.format,
    )
