#!/usr/bin/env python3
"""
环境诊断脚本 - 检查 Isaac Lab 和相关依赖是否正确安装
用于排查 ModuleNotFoundError: No module named 'isaaclab' 等问题
"""

import sys
import subprocess
import os

def print_section(title):
    """打印分节标题"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def check_python_version():
    """检查 Python 版本"""
    print_section("Python 版本检查")
    version = sys.version_info
    print(f"Python 版本: {version.major}.{version.minor}.{version.micro}")
    if version.major == 3 and version.minor == 10:
        print("✓ Python 版本符合要求 (3.10)")
        return True
    else:
        print(f"⚠ 警告: 推荐使用 Python 3.10，当前为 {version.major}.{version.minor}")
        return False

def check_module(module_name, import_name=None):
    """检查模块是否可以导入"""
    if import_name is None:
        import_name = module_name
    
    try:
        __import__(import_name)
        print(f"✓ {module_name} 已安装")
        return True
    except ImportError as e:
        print(f"✗ {module_name} 未安装或无法导入")
        print(f"  错误信息: {e}")
        return False

def check_isaac_lab():
    """检查 Isaac Lab 安装"""
    print_section("Isaac Lab 检查")
    
    # 检查 isaaclab 模块
    isaaclab_ok = check_module("isaaclab", "isaaclab")
    
    if isaaclab_ok:
        try:
            from isaaclab.app import AppLauncher
            print("✓ isaaclab.app.AppLauncher 可以正常导入")
            
            # 尝试获取版本信息
            try:
                import isaaclab
                if hasattr(isaaclab, '__version__'):
                    print(f"  Isaac Lab 版本: {isaaclab.__version__}")
            except:
                pass
                
            return True
        except ImportError as e:
            print(f"✗ isaaclab.app.AppLauncher 导入失败")
            print(f"  错误信息: {e}")
            return False
    else:
        return False

def check_project_modules():
    """检查项目相关模块"""
    print_section("项目模块检查")
    
    modules = [
        ("legged_lab", "legged_lab"),
        ("rsl_rl", "rsl_rl"),
    ]
    
    results = []
    for name, import_name in modules:
        results.append(check_module(name, import_name))
    
    return all(results)

def check_conda_env():
    """检查 conda 环境信息"""
    print_section("Conda 环境信息")
    
    try:
        env_name = os.environ.get('CONDA_DEFAULT_ENV', '未知')
        print(f"当前 Conda 环境: {env_name}")
        
        # 检查是否在 conda 环境中
        if env_name != '未知':
            print(f"✓ 正在使用 Conda 环境: {env_name}")
        else:
            print("⚠ 警告: 可能不在 Conda 环境中")
            
        return True
    except Exception as e:
        print(f"✗ 无法获取 Conda 环境信息: {e}")
        return False

def check_pip_packages():
    """检查关键 pip 包"""
    print_section("关键依赖包检查")
    
    packages = [
        "torch",
        "numpy",
        "gym",
    ]
    
    results = []
    for pkg in packages:
        results.append(check_module(pkg, pkg))
    
    return all(results)

def print_installation_guide():
    """打印安装指南"""
    print_section("安装指南")
    print("""
根据诊断结果，如果 isaaclab 未安装，请按以下步骤操作：

1. 安装 Isaac Lab (必需)
   Isaac Lab 不能通过 pip 直接安装，需要按照官方指南安装：
   
   访问: https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html
   
   推荐使用 Conda 安装方式：
   - 克隆 Isaac Lab 仓库
   - 按照官方文档创建 conda 环境并安装
   - 确保在安装 Isaac Lab 的 conda 环境中运行本项目

2. 安装项目依赖
   在已安装 Isaac Lab 的 conda 环境中执行：
   
   cd TienKung-Lab
   pip install -e .
   
   cd rsl_rl
   pip install -e .

3. 验证安装
   运行以下命令验证：
   
   python legged_lab/scripts/train.py --task=g1_dwaq --headless --num_envs=64
   
注意：
- Isaac Lab 需要 NVIDIA GPU 和 CUDA 支持
- 确保 Isaac Sim 已正确安装（Isaac Lab 的依赖）
- 本项目需要 Isaac Lab 2.1.0 或更高版本
""")

def main():
    """主函数"""
    print("\n" + "="*60)
    print("  G1DWAQ_Lab 环境诊断工具")
    print("="*60)
    
    results = {
        "Python 版本": check_python_version(),
        "Conda 环境": check_conda_env(),
        "Isaac Lab": check_isaac_lab(),
        "项目模块": check_project_modules(),
        "基础依赖": check_pip_packages(),
    }
    
    print_section("诊断总结")
    for name, result in results.items():
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{name}: {status}")
    
    if not results["Isaac Lab"]:
        print("\n⚠ 关键问题: Isaac Lab 未正确安装！")
        print_installation_guide()
    elif not all(results.values()):
        print("\n⚠ 部分检查未通过，请查看上面的详细信息")
    else:
        print("\n✓ 所有检查通过！环境配置正确。")

if __name__ == "__main__":
    main()

