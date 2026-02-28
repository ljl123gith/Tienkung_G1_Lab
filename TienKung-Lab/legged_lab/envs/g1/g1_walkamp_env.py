from __future__ import annotations

import json
import os
from typing import List

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from isaaclab.utils.math import quat_apply, quat_conjugate

from legged_lab.envs.g1.g1_env import G1Env
from legged_lab.envs.g1.g1_walkamp_cfg import G1WalkAmpEnvCfg


class G1WalkAmpEnv(G1Env):
    """G1 AMP walk environment on rough terrain."""

    def __init__(self, cfg: G1WalkAmpEnvCfg, headless: bool):
        super().__init__(cfg, headless)

        self.cfg: G1WalkAmpEnvCfg
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
        self.amp_joint_ids, _ = self.robot.find_joints(name_keys=self.amp_joint_names, preserve_order=True)

        # End-effector body ids
        self.ankle_body_ids, _ = self.robot.find_bodies(
            name_keys=["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
        )
        self.elbow_body_ids, _ = self.robot.find_bodies(
            name_keys=["left_elbow_link", "right_elbow_link"], preserve_order=True
        )

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
        mujoco_indices = {name: i for i, name in enumerate(self._mujoco_dof_names_29)}
        self._mujoco_to_lab_idx_29 = torch.tensor(
            [mujoco_indices[name] for name in self._lab_dof_names_29], device=self.device, dtype=torch.long
        )

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

        # Reorder joints if the file is in MuJoCo order (recommended default)
        if getattr(self.cfg, "motion_dof_order", "mujoco") == "mujoco" and dof_count == 29:
            dof_pos = dof_pos.index_select(0, self._mujoco_to_lab_idx_29)
            dof_vel = dof_vel.index_select(0, self._mujoco_to_lab_idx_29)

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
        """Return AMP observation for discriminator training (52 dims)."""
        robot = self.robot
        body_state_w = robot.data.body_state_w
        root_state_w = robot.data.root_state_w
        root_pos = root_state_w[:, 0:3]
        root_quat = root_state_w[:, 3:7]

        dof_pos = robot.data.joint_pos[:, self.amp_joint_ids]
        dof_vel = robot.data.joint_vel[:, self.amp_joint_ids]

        # Hand positions in body frame
        left_elbow_pos = body_state_w[:, self.elbow_body_ids[0], :3]
        right_elbow_pos = body_state_w[:, self.elbow_body_ids[1], :3]
        left_hand_pos = left_elbow_pos - root_pos + quat_apply(
            body_state_w[:, self.elbow_body_ids[0], 3:7], self.left_hand_local_vec
        )
        right_hand_pos = right_elbow_pos - root_pos + quat_apply(
            body_state_w[:, self.elbow_body_ids[1], 3:7], self.right_hand_local_vec
        )
        left_hand_pos = quat_apply(quat_conjugate(root_quat), left_hand_pos)
        right_hand_pos = quat_apply(quat_conjugate(root_quat), right_hand_pos)

        # Foot positions in body frame
        left_foot_pos = body_state_w[:, self.ankle_body_ids[0], :3] - root_pos
        right_foot_pos = body_state_w[:, self.ankle_body_ids[1], :3] - root_pos
        left_foot_pos = quat_apply(quat_conjugate(root_quat), left_foot_pos)
        right_foot_pos = quat_apply(quat_conjugate(root_quat), right_foot_pos)

        return torch.cat(
            (
                dof_pos,
                dof_vel,
                left_hand_pos,
                right_hand_pos,
                left_foot_pos,
                right_foot_pos,
            ),
            dim=-1,
        )

    def step(self, actions: torch.Tensor):
        """Execute one environment step."""
        delayed_actions = self.action_buffer.compute(actions)

        clipped_actions = torch.clip(delayed_actions, -self.clip_actions, self.clip_actions).to(self.device)
        processed_actions = clipped_actions * self.action_scale + self.robot.data.default_joint_pos

        self.avg_feet_force_per_step = torch.zeros(
            self.num_envs, len(self.feet_cfg.body_ids), dtype=torch.float, device=self.device, requires_grad=False
        )
        self.avg_feet_speed_per_step = torch.zeros(
            self.num_envs, len(self.feet_cfg.body_ids), dtype=torch.float, device=self.device, requires_grad=False
        )

        for _ in range(self.cfg.sim.decimation):
            self.sim_step_counter += 1
            self.robot.set_joint_position_target(processed_actions)
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)
            
            # Calculate feet force and speed for rewards
            self.avg_feet_force_per_step += torch.norm(
                self.contact_sensor.data.net_forces_w[:, self.feet_cfg.body_ids, :3], dim=-1
            )
            self.avg_feet_speed_per_step += torch.norm(
                self.robot.data.body_lin_vel_w[:, self.feet_cfg.body_ids, :], dim=-1
            )
            
        self.avg_feet_force_per_step /= self.cfg.sim.decimation
        self.avg_feet_speed_per_step /= self.cfg.sim.decimation

        if not self.headless:
            self.sim.render()

        self.episode_length_buf += 1
        self._calculate_gait_para()
        self.command_generator.compute(self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        self.reset_buf, self.time_out_buf = self.check_reset()
        reward_buf = self.reward_manager.compute(self.step_dt)
        self.reset_env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset(self.reset_env_ids)

        actor_obs, critic_obs = self.compute_observations()
        self.extras["observations"] = {"critic": critic_obs}

        return actor_obs, reward_buf, self.reset_buf, self.extras

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
