"""
从完整的 pkl 文件中裁剪指定帧范围，保留所有字段（包括末端位置）
"""
import pickle
import numpy as np
import argparse
from pathlib import Path


def crop_pkl_file(input_pkl, output_pkl, start_frame, end_frame, verbose=True):
    """
    从 pkl 文件中裁剪指定帧范围，保留所有字段
    
    Args:
        input_pkl: 输入 pkl 文件路径
        output_pkl: 输出 pkl 文件路径
        start_frame: 起始帧索引（包含）
        end_frame: 结束帧索引（不包含，即 [start_frame, end_frame)）
        verbose: 是否打印详细信息
    """
    if verbose:
        print("=" * 80)
        print(f"📂 加载文件: {input_pkl}")
        print("=" * 80)
    
    # 加载原始数据
    with open(input_pkl, "rb") as f:
        data = pickle.load(f)
    
    total_frames = None
    for key in ['root_pos', 'dof_pos', 'root_rot']:
        if key in data and data[key] is not None:
            total_frames = data[key].shape[0]
            break
    
    if total_frames is None:
        raise ValueError("无法确定总帧数")
    
    # 验证帧范围
    if start_frame < 0:
        start_frame = 0
    if end_frame > total_frames:
        end_frame = total_frames
    if start_frame >= end_frame:
        raise ValueError(f"无效的帧范围: [{start_frame}, {end_frame})")
    
    if verbose:
        print(f"📊 总帧数: {total_frames}")
        print(f"📊 裁剪范围: [{start_frame}, {end_frame})")
        print(f"📊 输出帧数: {end_frame - start_frame}")
        print(f"\n📋 原始文件包含的键: {list(data.keys())}")
    
    # 创建输出数据字典
    output_data = {}
    
    # 需要裁剪的字段（数组类型）
    array_fields = [
        'root_pos', 'root_rot', 'dof_pos',
        'left_foot_height', 'right_foot_height',
        'left_foot_pos_root', 'right_foot_pos_root',
        'left_hand_pos_root', 'right_hand_pos_root',
        'local_body_pos'  # 如果存在的话
    ]
    
    # 需要保留的字段（非数组类型）
    scalar_fields = ['fps', 'link_body_list']
    
    # 裁剪数组字段
    for key in array_fields:
        if key in data:
            value = data[key]
            if value is not None and isinstance(value, np.ndarray):
                if value.shape[0] == total_frames:
                    # 裁剪第一维
                    if len(value.shape) == 1:
                        output_data[key] = value[start_frame:end_frame].copy()
                    elif len(value.shape) == 2:
                        output_data[key] = value[start_frame:end_frame, :].copy()
                    elif len(value.shape) == 3:
                        output_data[key] = value[start_frame:end_frame, :, :].copy()
                    else:
                        # 对于更高维的数组，只裁剪第一维
                        slices = [slice(start_frame, end_frame)] + [slice(None)] * (len(value.shape) - 1)
                        output_data[key] = value[tuple(slices)].copy()
                    
                    if verbose:
                        print(f"✅ {key}: {data[key].shape} -> {output_data[key].shape}")
                else:
                    if verbose:
                        print(f"⚠️  {key}: 帧数不匹配 ({value.shape[0]} != {total_frames})，跳过")
            elif value is None:
                output_data[key] = None
                if verbose:
                    print(f"ℹ️  {key}: None")
            else:
                if verbose:
                    print(f"⚠️  {key}: 不是数组类型 ({type(value)})，跳过")
        else:
            if verbose:
                print(f"❌ {key}: 键不存在")
    
    # 保留标量字段
    for key in scalar_fields:
        if key in data:
            output_data[key] = data[key]
            if verbose:
                if isinstance(data[key], np.ndarray):
                    print(f"✅ {key}: shape={data[key].shape}")
                else:
                    print(f"✅ {key}: {data[key]}")
        else:
            if verbose:
                print(f"❌ {key}: 键不存在")
    
    # 保存输出文件
    if verbose:
        print(f"\n💾 保存到: {output_pkl}")
    
    with open(output_pkl, "wb") as f:
        pickle.dump(output_data, f)
    
    if verbose:
        print("✅ 裁剪完成！")
        print(f"\n📊 输出文件包含的键: {list(output_data.keys())}")
        print(f"📊 输出文件各字段形状:")
        for key, value in output_data.items():
            if value is not None:
                if isinstance(value, np.ndarray):
                    print(f"   {key}: {value.shape}")
                else:
                    print(f"   {key}: {type(value)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="从 pkl 文件中裁剪指定帧范围，保留所有字段"
    )
    parser.add_argument("--input_pkl", type=str, required=True, help="输入 pkl 文件路径")
    parser.add_argument("--output_pkl", type=str, required=True, help="输出 pkl 文件路径")
    parser.add_argument("--start_frame", type=int, required=True, help="起始帧索引（包含）")
    parser.add_argument("--end_frame", type=int, required=True, help="结束帧索引（不包含）")
    parser.add_argument("--quiet", action="store_true", help="静默模式，不打印详细信息")
    args = parser.parse_args()
    
    crop_pkl_file(
        args.input_pkl,
        args.output_pkl,
        args.start_frame,
        args.end_frame,
        verbose=not args.quiet
    )

