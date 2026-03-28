# G1  拟人风格化行走 （改进版）

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.3.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![RSL_RL](https://img.shields.io/badge/RSL_RL-2.3.1-silver)](https://github.com/leggedrobotics/rsl_rl)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](LICENSE)

![2026-03-28_22-05](/home/ljl/ljl_for_RL/TienKung_G1_Lab/TienKung-Lab/2026-03-28_22-05.png)

项目简介

本项目展示了基于深度强化学习的 **G1 25自由度人形机器人拟人风格化行走** 的算法框架

![2026-03-28_22-08](/home/ljl/ljl_for_RL/TienKung_G1_Lab/TienKung-Lab/2026-03-28_22-08.png)



本项目基于论文的核心思想参照“HumanMimic: Learning Natural Locomotion and Transitions for Humanoid Robot via Wasserstein Adversarial Imitation”，具体内容包AMP的观察空间和改进辨别器算法。https://arxiv.org/abs/2309.14225

## 改进的文件包括:

```
Tienkung_G1_Lab/
├── IsaacLab/                    # Isaac Lab 框架和仿真环境
├── TienKung-Lab/                # 基于 Legged Lab 的训练代码
	├── legged_lab/ 
    	├── assets/ 
    		├── unitree/ 
    			unitree_g1_25.py  #改为25dof的机器人环境
    	├── envs/ 
    		├── g1/ 
    			├── g1_walkamp_cfg_25.py 
    			├── g1_env_25.py
	├── rsl_rl/  
    	├── rsl_rl/ 
    		├── algorithms/ 
    			├── amp_ppo_WGAN_GP.py  #升级判别器算法
    	├── modules	/
			├── discriminator.py
		├──utils/
			├── motion_loader_amp.py
```

### 环境配置

#### 1. 安装 Isaac Lab

请按照 [Isaac Lab 官方安装指南](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)安装。建议使用 conda 环境便于从终端调用 Python 脚本。

#### 2. 获取项目代码

```bash
# 克隆本项目
git clone https://github.com/liuyufei-nubot/Tienkung_G1_Lab.git
cd Tienkung_G1_Lab
```

#### 3. 安装依赖

```bash
# 安装 TienKung-Lab
cd TienKung-Lab
pip install -e .

# 安装 rsl_rl
cd TienKung-Lab/rsl_rl
pip install -e .

# 安装 Unitree SDK (用于实物部署)
cd ../unitree_sdk2_python
pip install -e .

# 返回项目根目录
cd ..
```

### 训练



```bash
cd TienKung-Lab

# 训练
python legged_lab/scripts/train.py --task=g1_walkamp_25 --headless --num_envs=4096 
```

**参数说明**：

- `--task`: 任务名称（`g1_dwaq`, `g1_rough` 等）
- `--headless`: 无图形界面运行（推荐用于训练）
- `--num_envs`: 并行环境数量（根据 GPU 显存调整）
- `--max_iterations`: 最大训练迭代数

## 核心技术细节

### G1 机器人配置

**自由度配置**：25 DOF

当前训练使用的是 **Unitree G1 25自由度版本**，基于 `g1_29dof_simple_collision.urdf` 模型。

简化手部的腕关节，重点在机器人的下肢运动。

### Actor观测空间

| 观测项 | 维度 |
|--------|------|
| 角速度 (body frame) | 3 |
| 重力投影 (body frame) | 3 |
| 速度命令 [vx, vy, yaw_rate] | 3 |
| 关节位置偏差 | 25 |
| 关节速度 | 25 |
| 上一步动作 | 25 |
| 步态相位 (可选) | 1 |
| **总计** | **84** |



### AMP观测空间

| 观测项        | 维度   |
| ------------- | ------ |
| 关节位置      | 25     |
| 根坐标速度    | 3      |
| 根坐标角速度  | 3      |
| 关节速度      | 25     |
| 腿部Z方向速度 | 2      |
| **总计**      | **58** |



## 功能特性

### ✅ 已实现

- [x] IsaacLab 仿真环境
- [x] 强化学习训练流程
- [x] 简单的自适应步态行走
- [ ] 速度范围还有待扩展。

项目在https://github.com/liuyufei-nubot/G1DWAQ_Lab  基础上进行扩展开发。
