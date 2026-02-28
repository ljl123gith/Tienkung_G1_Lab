"""
检查 pkl 文件的结构和字段
用法:
    python check_pkl_keys.py --pkl_path /path/to/file.pkl
    或
    python check_pkl_keys.py  # 使用默认路径
"""
import pickle
import numpy as np
import argparse
import os


def check_pkl_keys(pkl_path):
    """检查 pkl 文件的所有键和字段信息"""
    if not os.path.exists(pkl_path):
        print(f"❌ 文件不存在: {pkl_path}")
        return
    
    print("=" * 80)
    print(f"📁 检查文件: {pkl_path}")
    print("=" * 80)
    
    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"❌ 读取文件时出错: {type(e).__name__}: {e}")
        return
    
    print("\n" + "=" * 80)
    print("📋 PKL 文件中的所有键:")
    print("=" * 80)
    
    for key in data.keys():
        value = data[key]
        if value is None:
            print(f"{key}: None")
        elif isinstance(value, np.ndarray):
            print(f"{key}: shape={value.shape}, dtype={value.dtype}")
            if value.size > 0:
                print(f"    范围: [{np.min(value):.6f}, {np.max(value):.6f}]")
        else:
            print(f"{key}: {type(value)} = {value}")
    
    print("\n" + "=" * 80)
    print("🔍 检查可能的字段名变体:")
    print("=" * 80)
    
    # 检查可能的字段名
    possible_names = {
        "脚部高度": [
            "left_foot_height", "right_foot_height", "foot_height",
            "l_foot_height", "r_foot_height", "left_foot_z", "right_foot_z"
        ],
        "脚部位置(root)": [
            "left_foot_pos_root", "right_foot_pos_root", "foot_pos_root",
            "l_foot_pos_root", "r_foot_pos_root", "left_foot_pos", "right_foot_pos"
        ],
        "手部位置(root)": [
            "left_hand_pos_root", "right_hand_pos_root", "hand_pos_root",
            "l_hand_pos_root", "r_hand_pos_root", "left_hand_pos", "right_hand_pos"
        ]
    }
    
    for category, names in possible_names.items():
        print(f"\n{category}:")
        found = []
        missing = []
        for name in names:
            if name in data and data[name] is not None:
                if isinstance(data[name], np.ndarray):
                    print(f"  ✅ 找到: {name} -> shape={data[name].shape}")
                else:
                    print(f"  ✅ 找到: {name} -> {type(data[name])}")
                found.append(name)
            else:
                missing.append(name)
        
        if not found:
            print("  ❌ 未找到任何字段")
            # 检查是否有类似的键
            similar = []
            for name in names:
                for k in data.keys():
                    if any(part in k.lower() for part in name.split('_') if len(part) > 2):
                        if k not in similar:
                            similar.append(k)
            if similar:
                print(f"  ⚠️  但找到类似的键: {similar}")
    
    # 统计信息
    print("\n" + "=" * 80)
    print("📊 数据统计:")
    print("=" * 80)
    
    # 确定总帧数
    total_frames = None
    for key in ['root_pos', 'dof_pos', 'root_rot']:
        if key in data and data[key] is not None and isinstance(data[key], np.ndarray):
            total_frames = data[key].shape[0]
            break
    
    if total_frames:
        fps = data.get('fps', 30)
        duration = total_frames / fps
        duration_min = duration / 60
        print(f"总帧数: {total_frames}")
        print(f"帧率: {fps} FPS")
        print(f"时长: {duration:.2f} 秒 ({duration_min:.2f} 分钟)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="检查 pkl 文件的结构和字段",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 检查指定文件
  python check_pkl_keys.py --pkl_path /path/to/file.pkl
  
  # 使用默认路径（如果脚本中有硬编码路径）
  python check_pkl_keys.py
        """
    )
    parser.add_argument(
        "--pkl_path",
        type=str,
        default=None,
        help="pkl 文件路径（如果不提供，将使用脚本中的默认路径）"
    )
    args = parser.parse_args()
    
    # 如果没有提供路径，使用默认路径
    if args.pkl_path is None:
        default_path = (
            "/home/ljl/ljl_for_RL/G1DWAQ_Lab/TienKung-Lab/legged_lab/"
            "envs/g1/datasets_ljl/motion_visualization/walk_0206_11_50_630.pkl"
        )
        print(f"⚠️  未提供 --pkl_path 参数，使用默认路径: {default_path}")
        pkl_path = default_path
    else:
        pkl_path = args.pkl_path
    
    check_pkl_keys(pkl_path)

