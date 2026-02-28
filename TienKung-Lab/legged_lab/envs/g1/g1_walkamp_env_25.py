from __future__ import annotations

import json
import os
from typing import List

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from isaaclab.utils.math import quat_apply, quat_conjugate, euler_xyz_from_quat

from legged_lab.envs.g1.g1_env_25 import G1Env_25
from legged_lab.envs.g1.g1_walkamp_cfg_25 import G1WalkAmpEnvCfg_25


class G1WalkAmpEnv_25(G1Env_25):
    """G1 AMP walk environment on rough terrain."""

    def __init__(self, cfg: G1WalkAmpEnvCfg_25, headless: bool):
        super().__init__(cfg, headless)

        self.cfg: G1WalkAmpEnvCfg_25
        self.reset_env_ids = torch.zeros(0, dtype=torch.long, device=self.device)

        # AMP joint subset (20 DoF)
        self.amp_joint_names = [
            "left_hip_yaw_joint",
            "left_hip_roll_joint",
            "left_hip_pitch_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_hip_yaw_joint",
            "right_hip_roll_joint",
            "right_hip_pitch_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
        ]
        # AMP joint subset (25 DoF)
        self.amp_joint_names_25 = [
            "left_hip_yaw_joint",
            "left_hip_roll_joint",
            "left_hip_pitch_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint", #6
            "right_hip_yaw_joint",
            "right_hip_roll_joint",
            "right_hip_pitch_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint", #6
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",  #3
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint", #5
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint", #5
        ]



        self.amp_joint_ids, _ = self.robot.find_joints(name_keys=self.amp_joint_names_25, preserve_order=True)

        # End-effector body ids
        self.ankle_body_ids, _ = self.robot.find_bodies(
            name_keys=["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
        )
        # 肘部刚体 ID（用于上肢摆动等观测）
        self.elbow_body_ids, _ = self.robot.find_bodies(
            name_keys=["left_elbow_link", "right_elbow_link"], preserve_order=True
        )
        # 手腕 roll 关节对应的刚体 ID：
        # 在 g1_25dof.urdf / g1_29dof.xml 中，关节名为 left_wrist_roll_joint/right_wrist_roll_joint，
        # 对应的 link 名为 left_wrist_roll_link/right_wrist_roll_link
        self.wrist_roll_body_ids, _ = self.robot.find_bodies(
            name_keys=["left_wrist_roll_link", "right_wrist_roll_link"], preserve_order=True
        )
        # 调试日志：只打印一次，确认找到的 ID 是否正确
        if not hasattr(self, "_debug_printed_wrist_ids"):
            print(f"[DEBUG][G1WalkAmpEnv_25] wrist_roll_body_ids = {self.wrist_roll_body_ids}")
            self._debug_printed_wrist_ids = True

        self.left_hand_local_vec = torch.tensor([0.0, 0.0, -0.3], device=self.device).repeat((self.num_envs, 1))
        self.right_hand_local_vec = torch.tensor([0.0, 0.0, -0.3], device=self.device).repeat((self.num_envs, 1))

        # Joint order mapping for motion_visualization playback
        # Many GMR/MoCap exports are in MuJoCo joint order; Isaac Lab uses a different 29-DoF order.
        self._mujoco_dof_names_29 = [
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ]
        self._lab_dof_names_29 = [
            "left_hip_pitch_joint",
            "right_hip_pitch_joint",
            "waist_yaw_joint",
            "left_hip_roll_joint",
            "right_hip_roll_joint",
            "waist_roll_joint",
            "left_hip_yaw_joint",
            "right_hip_yaw_joint",
            "waist_pitch_joint",
            "left_knee_joint",
            "right_knee_joint",
            "left_shoulder_pitch_joint",
            "right_shoulder_pitch_joint",
            "left_ankle_pitch_joint",
            "right_ankle_pitch_joint",
            "left_shoulder_roll_joint",
            "right_shoulder_roll_joint",
            "left_ankle_roll_joint",
            "right_ankle_roll_joint",
            "left_shoulder_yaw_joint",
            "right_shoulder_yaw_joint",
            "left_elbow_joint",
            "right_elbow_joint",
            "left_wrist_roll_joint",
            "right_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "right_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            "right_wrist_yaw_joint",
        ]
        self._mujoco_dof_names_25 = [
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            # "left_wrist_pitch_joint",
            # "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            # "right_wrist_pitch_joint",
            # "right_wrist_yaw_joint",
        ]
        self._lab_dof_names_25 = [
            "left_hip_pitch_joint",
            "right_hip_pitch_joint",
            "waist_yaw_joint",
            "left_hip_roll_joint",
            "right_hip_roll_joint",
            "waist_roll_joint",
            "left_hip_yaw_joint",
            "right_hip_yaw_joint",
            "waist_pitch_joint",
            "left_knee_joint",
            "right_knee_joint",
            "left_shoulder_pitch_joint",
            "right_shoulder_pitch_joint",
            "left_ankle_pitch_joint",
            "right_ankle_pitch_joint",
            "left_shoulder_roll_joint",
            "right_shoulder_roll_joint",
            "left_ankle_roll_joint",
            "right_ankle_roll_joint",
            "left_shoulder_yaw_joint",
            "right_shoulder_yaw_joint",
            "left_elbow_joint",
            "right_elbow_joint",
            "left_wrist_roll_joint",
            "right_wrist_roll_joint",
            # "left_wrist_pitch_joint",
            # "right_wrist_pitch_joint",
            # "left_wrist_yaw_joint",
            # "right_wrist_yaw_joint",
        ]

        mujoco_indices = {name: i for i, name in enumerate(self._mujoco_dof_names_29)}
        self._mujoco_to_lab_idx_29 = torch.tensor(
            [mujoco_indices[name] for name in self._lab_dof_names_29], device=self.device, dtype=torch.long
        )

        # 25-DoF: MuJoCo → Lab 关节索引映射（用于 GMR 64D / 25DoF 数据）
        mujoco_indices_25 = {name: i for i, name in enumerate(self._mujoco_dof_names_25)}
        self._mujoco_to_lab_idx_25 = torch.tensor(
            [mujoco_indices_25[name] for name in self._lab_dof_names_25],
            device=self.device,
            dtype=torch.long,
        )

        # 调试日志：只打印一次，确认 25-DoF 映射是否按预期建立
        if not hasattr(self, "_debug_printed_mujoco_map_25"):
            print(
                "[DEBUG][G1WalkAmpEnv_25] _mujoco_to_lab_idx_25 = "
                f"{self._mujoco_to_lab_idx_25.cpu().tolist()}"
            )
            self._debug_printed_mujoco_map_25 = True

        # Load motion visualization data (for play_amp_animation.py)
        self._visual_frames: List[np.ndarray] = []
        self._visual_frame_durations: List[float] = []
        self._load_motion_visualization_files(self.cfg.motion_visualization_files)
        self._visual_motion_index = (
            min(max(self.cfg.motion_visualization_index, 0), len(self._visual_frames) - 1)
            if self._visual_frames
            else 0
        )
        if self._visual_frames:
            self.motion_len = len(self._visual_frames[self._visual_motion_index])
            self._visual_frame_duration = self._visual_frame_durations[self._visual_motion_index]
        else:
            self.motion_len = 0
            self._visual_frame_duration = self.cfg.motion_frame_duration

    def init_buffers(self):
        """Initialize buffers, ensuring gait params are set before observation computation."""
        # Init gait parameter - required for compute_current_observations -> init_obs_buffer
        self.gait_phase = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)
        self.gait_cycle = torch.full(
            (self.num_envs,), self.cfg.gait.gait_cycle, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.phase_ratio = torch.tensor(
            [self.cfg.gait.gait_air_ratio_l, self.cfg.gait.gait_air_ratio_r], dtype=torch.float, device=self.device
        ).repeat(self.num_envs, 1)
        self.phase_offset = torch.tensor(
            [self.cfg.gait.gait_phase_offset_l, self.cfg.gait.gait_phase_offset_r],
            dtype=torch.float,
            device=self.device,
        ).repeat(self.num_envs, 1)

        # Call super which calls init_obs_buffer -> compute_current_observations
        super().init_buffers()

        # Helper buffers for gait rewards (requires feet_cfg initialized in super().init_buffers)
        self.avg_feet_force_per_step = torch.zeros(
            self.num_envs, len(self.feet_cfg.body_ids), dtype=torch.float, device=self.device, requires_grad=False
        )
        self.avg_feet_speed_per_step = torch.zeros(
            self.num_envs, len(self.feet_cfg.body_ids), dtype=torch.float, device=self.device, requires_grad=False
        )

    def _load_motion_visualization_files(self, file_paths: list[str]) -> None:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        for file_path in file_paths:
            resolved_path = file_path
            if not os.path.isabs(resolved_path):
                if file_path.startswith("legged_lab/"):
                    resolved_path = os.path.join(repo_root, file_path)
                else:
                    resolved_path = os.path.join(base_dir, file_path)
            with open(resolved_path, "r") as f:
                motion_json = json.load(f)
            frames = np.asarray(motion_json["Frames"], dtype=np.float32)
            frame_duration = float(motion_json.get("FrameDuration", self.cfg.motion_frame_duration))
            self._visual_frames.append(frames)
            self._visual_frame_durations.append(frame_duration)

    def _get_visual_frame_at_time(self, time_s: float) -> np.ndarray:
        if not self._visual_frames:
            raise RuntimeError("No motion visualization files loaded.")
        frames = self._visual_frames[self._visual_motion_index]
        frame_duration = self._visual_frame_durations[self._visual_motion_index]
        if frame_duration <= 0:
            frame_duration = self.cfg.motion_frame_duration
        frame_idx = int(round(time_s / frame_duration))
        frame_idx = max(0, min(frame_idx, len(frames) - 1))
        return frames[frame_idx]

    def visualize_motion(self, time: float):
        """Update sim from visualization frame and return AMP obs."""
        frame = self._get_visual_frame_at_time(time)
        frame_len = frame.shape[0]

        # 兼容两种格式：
        # 1) 通用格式：len = 12 + 2 * N
        #    [0:3]  root_pos, [3:6] euler, [6:6+N] dof_pos,
        #    [6+N:9+N] root_lin_vel, [9+N:12+N] root_ang_vel,
        #    [12+N:12+2N] dof_vel
        # 2) GMR 64D 格式（当前 25-DoF Expert 使用）：
        #    [0:3]  root_pos
        #    [3:6]  euler
        #    [6:31] dof_pos (25)
        #    [31:34] root_lin_vel
        #    [34:37] root_ang_vel
        #    [37:62] dof_vel (25)
        #    [62:64] foot_heights(2)  ← 这里只在 AMP obs 中用，不参与动画重放
        if frame_len == 64 and self.robot.num_joints == 25:
            # 将 64 维帧重排为通用的 12 + 2*N (=62) 格式，丢弃最后两维脚高
            dof_count = 25
            new_len = 12 + 2 * dof_count  # 62
            frame_generic = np.zeros(new_len, dtype=frame.dtype)

            # 基座位置/姿态
            frame_generic[0:3] = frame[0:3]
            frame_generic[3:6] = frame[3:6]
            # 关节位置
            frame_generic[6 : 6 + dof_count] = frame[6:31]
            # 根速度
            frame_generic[6 + dof_count : 9 + dof_count] = frame[31:34]
            frame_generic[9 + dof_count : 12 + dof_count] = frame[34:37]
            # 关节速度
            frame_generic[12 + dof_count : 12 + 2 * dof_count] = frame[37:62]

            frame = frame_generic
            frame_len = frame.shape[0]

        dof_count = (frame_len - 12) // 2
        if frame_len != 12 + 2 * dof_count:
            raise ValueError(f"Invalid frame length: {frame_len}. Expected 12 + 2*N.")
        if dof_count != self.robot.num_joints:
            raise ValueError(
                f"Frame dof count {dof_count} != robot joints {self.robot.num_joints}. "
                "Check joint order/mapping for motion_visualization files."
            )

        device = self.device
        frame_t = torch.tensor(frame, device=device, dtype=torch.float32)
        root_pos = frame_t[0:3].clone()
        root_pos[2] += self.cfg.root_height_offset
        euler = frame_t[3:6].cpu().numpy()
        dof_pos = frame_t[6 : 6 + dof_count]
        root_lin_vel = frame_t[6 + dof_count : 9 + dof_count]
        root_ang_vel = frame_t[9 + dof_count : 12 + dof_count]
        dof_vel = frame_t[12 + dof_count : 12 + 2 * dof_count]

        # Reorder joints if the file is in MuJoCo joint order (recommended default for GMR)
        if getattr(self.cfg, "motion_dof_order", "mujoco") == "mujoco":
            if dof_count == 29:
                # 29-DoF 老数据: 使用 _mujoco_to_lab_idx_29
                dof_pos = dof_pos.index_select(0, self._mujoco_to_lab_idx_29)
                dof_vel = dof_vel.index_select(0, self._mujoco_to_lab_idx_29)
            elif dof_count == 25:
                # 25-DoF GMR 64D 数据: 使用 _mujoco_to_lab_idx_25
                dof_pos = dof_pos.index_select(0, self._mujoco_to_lab_idx_25)
                dof_vel = dof_vel.index_select(0, self._mujoco_to_lab_idx_25)
            else:
                # 其他 dof_count 不做重排，仅打印一次警告，方便后续扩展
                if not hasattr(self, "_debug_printed_motion_dof_warning"):
                    print(
                        f"[WARN][G1WalkAmpEnv_25] motion_dof_order='mujoco' "
                        f"but unsupported dof_count={dof_count} for visualize_motion."
                    )
                    self._debug_printed_motion_dof_warning = True

        quat_xyzw = Rotation.from_euler("XYZ", euler, degrees=False).as_quat()
        quat_wxyz = torch.tensor(
            [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=torch.float32, device=device
        )

        dof_pos_all = dof_pos.unsqueeze(0).repeat(self.num_envs, 1)
        dof_vel_all = dof_vel.unsqueeze(0).repeat(self.num_envs, 1)
        self.robot.write_joint_position_to_sim(dof_pos_all)
        self.robot.write_joint_velocity_to_sim(dof_vel_all)

        root_state = torch.zeros((self.num_envs, 13), device=device)
        root_state[:, 0:3] = root_pos.unsqueeze(0).repeat(self.num_envs, 1)
        root_state[:, 3:7] = quat_wxyz.unsqueeze(0).repeat(self.num_envs, 1)
        root_state[:, 7:10] = root_lin_vel.unsqueeze(0).repeat(self.num_envs, 1)
        root_state[:, 10:13] = root_ang_vel.unsqueeze(0).repeat(self.num_envs, 1)
        env_ids = torch.arange(self.num_envs, device=device)
        self.robot.write_root_state_to_sim(root_state, env_ids)

        if not self.headless:
            self.sim.render()
        self.sim.step()
        self.scene.update(dt=self.step_dt)

        return self.get_amp_obs_for_expert_trans()

    def get_amp_obs_for_expert_trans(self) -> torch.Tensor:
        # """
        # Return AMP observation matching the 64D expert format.

        # 这里严格对齐 `gmr_data_conversion_copy.py` 里 64 维数据的拼接顺序：

        #     [0:3]   root_pos (world)
        #     [3:6]   root_euler XYZ (from quat, world)
        #     [6:31]  dof_pos (25 joints)
        #     [31:34] root_lin_vel (world)
        #     [34:37] root_ang_vel (world)
        #     [37:62] dof_vel (25 joints)
        #     [62:63] left_foot_height (world z)
        #     [63:64] right_foot_height (world z)

        # 这样专家 txt（64 维）和环境在线生成的 AMP 观测可以一一对应，方便判别器学习。
        # """
        # robot = self.robot
        # body_state_w = robot.data.body_state_w
        # root_state_w = robot.data.root_state_w

        # # 1) 根节点世界坐标 (3)
        # root_pos = root_state_w[:, 0:3]

        # # 2) 根节点欧拉角 XYZ (3) —— 与脚本中的 as_euler('XYZ') 保持一致
        # root_quat = root_state_w[:, 3:7]  # isaaclab 默认是 wxyz
        # roll, pitch, yaw = euler_xyz_from_quat(root_quat)
        # root_euler = torch.stack((roll, pitch, yaw), dim=-1)

        # # 3) 关节位置 (25) —— 使用 AMP 关节子集顺序
        # dof_pos = robot.data.joint_pos[:, self.amp_joint_ids]

        # # 4) 根节点世界系线速度 (3)
        # # gmr_data_conversion_copy.py 中是直接在世界坐标下对 root_pos 做差分，因此这里也使用 *_w
        # root_lin_vel = robot.data.root_lin_vel_w

        # # 5) 根节点世界系角速度 (3)
        # root_ang_vel = robot.data.root_ang_vel_w

        # # 6) 关节速度 (25)
        # dof_vel = robot.data.joint_vel[:, self.amp_joint_ids]

        # # 7) 脚部相对地面的高度 (2) —— 直接取世界坐标 z 分量
        # left_foot_height = body_state_w[:, self.ankle_body_ids[0], 2:3]
        # right_foot_height = body_state_w[:, self.ankle_body_ids[1], 2:3]

        # # 拼接 AMP 观测，严格按照 64 维顺序
        # amp_obs = torch.cat(
        #     (
        #         root_pos,          # [0:3]
        #         root_euler,        # [3:6]
        #         dof_pos,           # [6:31]
        #         root_lin_vel,      # [31:34]
        #         root_ang_vel,      # [34:37]
        #         dof_vel,           # [37:62]
        #         left_foot_height,  # [62:63]
        #         right_foot_height, # [63:64]
        #     ),
        #     dim=-1,
        # )

        # # 只在第一次调用时打印一次维度，确认与专家数据一致
        # if not hasattr(self, "_debug_printed_amp_dim"):
        #     print(f"[DEBUG][AMP] env get_amp_obs_for_expert_trans dim = {amp_obs.shape[1]}")
        #     self._debug_printed_amp_dim = True

        # return amp_obs
        """Return AMP observation strictly matching the 58D local expert format."""
        robot = self.robot
        body_state_w = robot.data.body_state_w
        
        # 1. 关节位置 (25)
        dof_pos = robot.data.joint_pos[:, self.amp_joint_ids]
        
        # 2. 根节点线速度 - 【极其重要：必须使用 _b (基座局部系)】
        root_lin_vel_b = robot.data.root_lin_vel_b 
        
        # 3. 根节点角速度 - 【极其重要：必须使用 _b (基座局部系)】
        root_ang_vel_b = robot.data.root_ang_vel_b
        
        # 4. 关节速度 (25)
        dof_vel = robot.data.joint_vel[:, self.amp_joint_ids]

        # 5. 脚部相对地面的高度 (2)
        # 提示：高度(Z)本身是旋转不变的，可以继续用 _w 坐标系下的 Z 轴
        left_foot_height = body_state_w[:, self.ankle_body_ids[0], 2:3]
        right_foot_height = body_state_w[:, self.ankle_body_ids[1], 2:3]

        # 拼接这 58 维：完全不包含绝对的 x,y,z 和 roll,pitch,yaw
        amp_obs = torch.cat(
            (
                dof_pos,            # [0:25]
                root_lin_vel_b,     # [25:28] 基座坐标系线速度
                root_ang_vel_b,     # [28:31] 基座坐标系角速度
                dof_vel,            # [31:56]
                left_foot_height,   # [56:57]
                right_foot_height,  # [57:58]
            ),
            dim=-1,
        )

        # 打印验证维度
        if not hasattr(self, "_debug_printed_amp_dim"):
            print(f"[DEBUG][AMP] env get_amp_obs_for_expert_trans dim = {amp_obs.shape[1]}")
            self._debug_printed_amp_dim = True

        return amp_obs
        

    def step(self, actions: torch.Tensor):
        """
        Execute one environment step and attach AMP observations.

        这里**不重新实现整套仿真循环**，而是完全复用 `G1Env_25.step` 的逻辑：
        - 由基类完成：动作处理 → 物理 decimation → 观测计算 → 奖励与 reset 逻辑
        - 我们只在返回的 `infos` 里额外挂上 AMP 需要的观测，用于 rsl_rl 的 AMP 判别器。
        这样可以避免和 IsaacLab / Legged-Lab 的内部流程打架，同时保证接口兼容
        `AmpOnPolicyRunner.learn()` 里期望的 4 元组：obs, rewards, dones, infos。
        """
        # 1) 调用基类 step：完成标准 RL 环节（包含 gait phase / feet 力等内部维护）
        obs, rewards, dones, infos = super().step(actions)

        # 2) 在 infos 里塞入 AMP 观测，供 AMP-PP0 算法和判别器使用
        if "observations" not in infos:
            infos["observations"] = {}

        amp_obs = self.get_amp_obs_for_expert_trans()
        # 调试一次 AMP 观测形状，方便对齐专家数据维度
        if not hasattr(self, "_debug_printed_step_amp_dim"):
            print(f"[DEBUG][AMP][Env.step] attaching amp_obs with dim={amp_obs.shape[1]}")
            self._debug_printed_step_amp_dim = True

        infos["observations"]["amp"] = amp_obs

        return obs, rewards, dones, infos


    # def step(self, actions: torch.Tensor):
    #     """Execute one environment step."""
    #     delayed_actions = self.action_buffer.compute(actions)

    #     clipped_actions = torch.clip(delayed_actions, -self.clip_actions, self.clip_actions).to(self.device)
    #     processed_actions = clipped_actions * self.action_scale + self.robot.data.default_joint_pos

    #     self.avg_feet_force_per_step = torch.zeros(
    #         self.num_envs, len(self.feet_cfg.body_ids), dtype=torch.float, device=self.device, requires_grad=False
    #     )
    #     self.avg_feet_speed_per_step = torch.zeros(
    #         self.num_envs, len(self.feet_cfg.body_ids), dtype=torch.float, device=self.device, requires_grad=False
    #     )

    #     for _ in range(self.cfg.sim.decimation):
    #         self.sim_step_counter += 1
    #         self.robot.set_joint_position_target(processed_actions)
    #         self.scene.write_data_to_sim()
    #         self.sim.step(render=False)
    #         self.scene.update(dt=self.physics_dt)
            
    #         # Calculate feet force and speed for rewards
    #         self.avg_feet_force_per_step += torch.norm(
    #             self.contact_sensor.data.net_forces_w[:, self.feet_cfg.body_ids, :3], dim=-1
    #         )
    #         self.avg_feet_speed_per_step += torch.norm(
    #             self.robot.data.body_lin_vel_w[:, self.feet_cfg.body_ids, :], dim=-1
    #         )
            
    #     self.avg_feet_force_per_step /= self.cfg.sim.decimation
    #     self.avg_feet_speed_per_step /= self.cfg.sim.decimation

    #     if not self.headless:
    #         self.sim.render()

    #     self.episode_length_buf += 1
    #     self._calculate_gait_para()
    #     self.command_generator.compute(self.step_dt)
    #     if "interval" in self.event_manager.available_modes:
    #         self.event_manager.apply(mode="interval", dt=self.step_dt)

    #     self.reset_buf, self.time_out_buf = self.check_reset()
    #     reward_buf = self.reward_manager.compute(self.step_dt)
    #     self.reset_env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
    #     self.reset(self.reset_env_ids)

    #     actor_obs, critic_obs = self.compute_observations()
    #     self.extras["observations"] = {"critic": critic_obs}

    #     return actor_obs, reward_buf, self.reset_buf, self.extras

    def _calculate_gait_para(self) -> None:
        """
        Update gait phase parameters based on simulation time and offset.
        """
        t = self.episode_length_buf * self.step_dt / self.gait_cycle
        self.gait_phase[:, 0] = (t + self.phase_offset[:, 0]) % 1.0
        self.gait_phase[:, 1] = (t + self.phase_offset[:, 1]) % 1.0

    def compute_current_observations(self):
        """Compute current step observations including gait phase."""
        # Get base observations from G1Env
        current_actor_obs, current_critic_obs = super().compute_current_observations()
        
        # Create Gait observations
        gait_obs = torch.cat(
            [
                torch.sin(2 * torch.pi * self.gait_phase),  # 2
                torch.cos(2 * torch.pi * self.gait_phase),  # 2
                self.phase_ratio,  # 2
            ],
            dim=-1,
        )
        
        # Append gait obs to actor obs
        new_actor_obs = torch.cat([current_actor_obs, gait_obs], dim=-1)
        
        # Verify if separate critic obs needs gait. 
        # In G1RoughEnvCfg (which G1WalkAmpEnvCfg inherits/modifies), we have asymmetric actor-critic.
        # But G1Env.compute_current_observations builds critic_obs by concatenating actor_obs + extras.
        # However, it concatenated the OLD actor_obs.
        # We need to rebuild or append to critic obs.
        
        # G1Env.compute_current_observations:
        # critic_obs_list = [current_actor_obs, root_lin_vel * scales, feet_contact, (privileged...)]
        # So super() returned critic_obs used the OLD actor_obs.
        # If we just want to add gait to critic obs, we can append it?
        # OR we can assume that since actor_obs is part of critic_obs, we should reconstruct critic_obs?
        # Reconstructing is hard because we don't return all intermediate parts from super.
        
        # EASIER: Just append gait_obs to critic_obs as well, assuming specific order doesn't fail everything.
        # Ideally, critic should see everything actor sees + extras.
        # If super's critic_obs = [old_actor, extras], then [new_actor, extras] is desired.
        # But we have [old_actor, extras]. 
        # new_actor = [old_actor, gait].
        # So we want [old_actor, gait, extras].
        # But getting [old_actor, extras, gait] is also fine for dense nets usually.
        # Let's just append gait to critic_obs too.
        
        new_critic_obs = torch.cat([current_critic_obs, gait_obs], dim=-1)
        
        return new_actor_obs, new_critic_obs
