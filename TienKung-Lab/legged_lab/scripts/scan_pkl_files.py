import pickle
import numpy as np
import os

def scan_pkl_file(file_path):
    """详细扫描 pkl 文件的结构和内容"""
    print("=" * 80)
    print(f"📁 扫描文件: {file_path}")
    print("=" * 80)
    
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return
    
    # 检查文件大小
    file_size = os.path.getsize(file_path)
    print(f"📦 文件大小: {file_size / 1024:.2f} KB")
    
    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        
        print(f"\n🔑 数据类型: {type(data)}")
        
        if isinstance(data, dict):
            print(f"📋 字典键数量: {len(data)}")
            print(f"📋 所有键: {list(data.keys())}")
            
            print("\n" + "-" * 80)
            print("📊 详细字段信息:")
            print("-" * 80)
            
            for key, value in data.items():
                print(f"\n🔸 {key}:")
                if value is None:
                    print(f"   类型: None")
                elif isinstance(value, np.ndarray):
                    print(f"   类型: numpy.ndarray")
                    print(f"   Shape: {value.shape}")
                    print(f"   Dtype: {value.dtype}")
                    print(f"   内存大小: {value.nbytes / 1024:.2f} KB")
                    if value.size > 0:
                        print(f"   最小值: {np.min(value):.6f}")
                        print(f"   最大值: {np.max(value):.6f}")
                        print(f"   平均值: {np.mean(value):.6f}")
                    if value.size <= 20:
                        print(f"   数据预览: {value}")
                    else:
                        print(f"   前5个值: {value.flat[:5]}")
                        print(f"   后5个值: {value.flat[-5:]}")
                elif isinstance(value, (list, tuple)):
                    print(f"   类型: {type(value).__name__}")
                    print(f"   长度: {len(value)}")
                    if len(value) > 0:
                        print(f"   第一个元素类型: {type(value[0])}")
                        if len(value) <= 10:
                            print(f"   内容: {value}")
                        else:
                            print(f"   前3个元素: {value[:3]}")
                            print(f"   后3个元素: {value[-3:]}")
                elif isinstance(value, (int, float, str, bool)):
                    print(f"   类型: {type(value).__name__}")
                    print(f"   值: {value}")
                else:
                    print(f"   类型: {type(value)}")
                    print(f"   值: {str(value)[:100]}...")
        
        elif isinstance(data, np.ndarray):
            print(f"📊 数组信息:")
            print(f"   Shape: {data.shape}")
            print(f"   Dtype: {data.dtype}")
            print(f"   内存大小: {data.nbytes / 1024:.2f} KB")
        
        else:
            print(f"📊 其他类型数据: {type(data)}")
            print(f"   内容预览: {str(data)[:200]}...")
        
        # 检查是否有我们期望的字段
        print("\n" + "-" * 80)
        print("🔍 检查期望的字段:")
        print("-" * 80)
        
        expected_fields = {
            "基础字段": ["root_pos", "root_rot", "dof_pos", "fps"],
            "脚部高度": ["left_foot_height", "right_foot_height", "l_foot_height", "r_foot_height"],
            "脚部位置(root)": ["left_foot_pos_root", "right_foot_pos_root", "l_foot_pos_root", "r_foot_pos_root"],
            "手部位置(root)": ["left_hand_pos_root", "right_hand_pos_root", "l_hand_pos_root", "r_hand_pos_root"],
            "其他": ["local_body_pos", "link_body_list"]
        }
        
        if isinstance(data, dict):
            for category, fields in expected_fields.items():
                print(f"\n{category}:")
                found = []
                missing = []
                for field in fields:
                    if field in data and data[field] is not None:
                        found.append(field)
                    else:
                        missing.append(field)
                if found:
                    print(f"   ✅ 找到: {found}")
                if missing:
                    print(f"   ❌ 缺失: {missing}")
        
    except Exception as e:
        print(f"❌ 读取文件时出错: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    base_dir = "/home/ljl/ljl_for_RL/G1DWAQ_Lab/TienKung-Lab/legged_lab/envs/g1/datasets_ljl/motion_visualization"
    
    files = [
        "walk_0206_12.pkl",
        "walk_0206_11.pkl"
    ]
    
    for filename in files:
        file_path = os.path.join(base_dir, filename)
        scan_pkl_file(file_path)
        print("\n\n")

