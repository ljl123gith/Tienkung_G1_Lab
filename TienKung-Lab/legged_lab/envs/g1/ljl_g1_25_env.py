

import isaaclab.sim as sim_utils
import isaacsim.core.utils.torch as torch_utils  # type: ignore
import numpy as np
import torch
from isaaclab.assets.articulation import Articulation
from isaaclab.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.managers import EventManager, RewardManager
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.scene import InteractiveScene
from isaaclab.sensors import ContactSensor, RayCaster

from isaaclab.sim import PhysxCfg, SimulationContext
from isaaclab.utils.buffers import CircularBuffer, DelayBuffer
from isaaclab.utils.math import quat_apply, quat_conjugate
from scipy.spatial.transform import Rotation

# Import the configuration for G1 25DOF robot
from legged_lab.envs.g1.ljl_walk_25_cfg import G125WalkFlatEnvCfg
from legged_lab.utils.env_utils.scene import SceneCfg
from rsl_rl.env import VecEnv
from rsl_rl.utils import AMPLoaderDisplay


class G125Env(VecEnv):
    """Environment for G1 25DOF robot with walking tasks."""
    
    def __init__(
        self,
        cfg: G125WalkFlatEnvCfg,
        headless,
    ):
        """Initialize the G1 25DOF environment.
        
        Args:
            cfg: Environment configuration for G1 25DOF robot
            headless: Whether to run in headless mode (no visualization)
        """
        self.cfg: G125WalkFlatEnvCfg

        self.cfg = cfg
        self.headless = headless
        self.device = self.cfg.device
        self.physics_dt = self.cfg.sim.dt
        self.step_dt = self.cfg.sim.decimation * self.cfg.sim.dt
        self.num_envs = self.cfg.scene.num_envs
        self.seed(cfg.scene.seed)

        # Configure simulation parameters
        sim_cfg = sim_utils.SimulationCfg(
            device=cfg.device,
            dt=cfg.sim.dt,
            render_interval=cfg.sim.decimation,
            physx=PhysxCfg(gpu_max_rigid_patch_count=cfg.sim.physx.gpu_max_rigid_patch_count),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
        )
        self.sim = SimulationContext(sim_cfg)

        # Create the scene with robot and terrain
        scene_cfg = SceneCfg(config=cfg.scene, physics_dt=self.physics_dt, step_dt=self.step_dt)
        self.scene = InteractiveScene(scene_cfg)
        self.sim.reset()

        # Get references to scene entities
        self.robot: Articulation = self.scene["robot"]
        self.contact_sensor: ContactSensor = self.scene.sensors["contact_sensor"]

        # Initialize height scanner if enabled
        if self.cfg.scene.height_scanner.enable_height_scan:
            self.height_scanner: RayCaster = self.scene.sensors["height_scanner"]

        # Initialize LiDAR and Depth Camera Sensors if enabled
        if self.cfg.scene.lidar.enable_lidar:
            self.lidar: RayCaster = self.scene.sensors["lidar"]
        if self.cfg.scene.depth_camera.enable_depth_camera:
            self.depth_camera: TiledCamera = self.scene.sensors["depth_camera"]

        # Setup command generator for velocity commands
        command_cfg = UniformVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=self.cfg.commands.resampling_time_range,
            rel_standing_envs=self.cfg.commands.rel_standing_envs,
            rel_heading_envs=self.cfg.commands.rel_heading_envs,
            heading_command=self.cfg.commands.heading_command,
            heading_control_stiffness=self.cfg.commands.heading_control_stiffness,
            debug_vis=self.cfg.commands.debug_vis,
            ranges=self.cfg.commands.ranges,
        )
        self.command_generator = UniformVelocityCommand(cfg=command_cfg, env=self)
        self.reward_manager = RewardManager(self.cfg.reward, self)

        # Initialize buffers and reset environment
        self.init_buffers()

        env_ids = torch.arange(self.num_envs, device=self.device)
        self.event_manager = EventManager(self.cfg.domain_rand.events, self)
        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")
        self.reset(env_ids)

        # Load AMP motion data for visualization
        # self.amp_loader_display = AMPLoaderDisplay(
        #     motion_files=self.cfg.amp_motion_files_display, device=self.device, time_between_frames=self.physics_dt
        # )
        # self.motion_len = self.amp_loader_display.trajectory_num_frames[0]
        self._visual_frames = []           # List[torch.Tensor], 每个元素 shape=(T,64)
        self._visual_frame_durations = []  # 每个 motion 的 FrameDuration

        import json, os
        motion_files = self.cfg.amp_motion_files_display
        if isinstance(motion_files, str):
            motion_files = [motion_files]

        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))  # 可视情况调整
        for fp in motion_files:
            resolved = fp
            if not os.path.isabs(resolved):
                resolved = os.path.join(base_dir, fp)
            with open(resolved, "r") as f:
                motion_json = json.load(f)
            frames_np = np.asarray(motion_json["Frames"], dtype=np.float32)
            frames = torch.tensor(frames_np, device=self.device)
            self._visual_frames.append(frames)
            self._visual_frame_durations.append(float(motion_json.get("FrameDuration", self.physics_dt)))

        self._visual_motion_index = 0
        if self._visual_frames:
            self.motion_len = self._visual_frames[self._visual_motion_index].shape[0]
            self._visual_frame_duration = self._visual_frame_durations[self._visual_motion_index]
        else:
            self.motion_len = 0
            self._visual_frame_duration = self.physics_dt

    def _get_visual_frame_at_time(self, time_s: float) -> torch.Tensor:
        """按照时间戳从 64 维 GMR JSON 里取一帧。"""
        if not self._visual_frames:
            raise RuntimeError("No motion visualization files loaded for G1 25DOF.")
        frames = self._visual_frames[self._visual_motion_index]
        frame_duration = self._visual_frame_duration if self._visual_frame_duration > 0 else self.physics_dt
        frame_idx = int(round(time_s / frame_duration))
        frame_idx = max(0, min(frame_idx, frames.shape[0] - 1))
        return frames[frame_idx]

    def init_buffers(self):
        """Initialize all buffers and parameters for the environment."""
        self.extras = {}

        # Episode length configuration
        self.max_episode_length_s = self.cfg.scene.max_episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.step_dt)
        self.num_actions = self.robot.data.default_joint_pos.shape[1]
        self.clip_actions = self.cfg.normalization.clip_actions
        self.clip_obs = self.cfg.normalization.clip_observations

        # Action configuration with delay buffer
        self.action_scale = self.cfg.robot.action_scale
        self.action_buffer = DelayBuffer(
            self.cfg.domain_rand.action_delay.params["max_delay"], self.num_envs, device=self.device
        )
        self.action_buffer.compute(
            torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        )
        
        # Setup action delay if enabled
        if self.cfg.domain_rand.action_delay.enable:
            time_lags = torch.randint(
                low=self.cfg.domain_rand.action_delay.params["min_delay"],
                high=self.cfg.domain_rand.action_delay.params["max_delay"] + 1,
                size=(self.num_envs,),
                dtype=torch.int,
                device=self.device,
            )
            self.action_buffer.set_time_lag(time_lags, torch.arange(self.num_envs, device=self.device))

        # Configure robot body and contact sensors
        self.robot_cfg = SceneEntityCfg(name="robot")
        self.robot_cfg.resolve(self.scene)
        self.termination_contact_cfg = SceneEntityCfg(
            name="contact_sensor", body_names=self.cfg.robot.terminate_contacts_body_names
        )
        self.termination_contact_cfg.resolve(self.scene)
        self.feet_cfg = SceneEntityCfg(name="contact_sensor", body_names=self.cfg.robot.feet_body_names)
        self.feet_cfg.resolve(self.scene)

        # Find body IDs for feet and elbows (used for AMP observations)
        self.feet_body_ids, _ = self.robot.find_bodies(
            name_keys=["left_ankle_roll_link", "right_ankle_roll_link"],
            preserve_order=True
        )
        self.elbow_body_ids, _ = self.robot.find_bodies(
            name_keys=["left_elbow_link", "right_elbow_link"],
            preserve_order=True
        )
        
        # Find joint IDs for different body parts
        self.left_leg_ids, _ = self.robot.find_joints(
            name_keys=[
                "left_hip_pitch_joint",
                "left_hip_roll_joint",
                "left_hip_yaw_joint",  
                "left_knee_joint",
                "left_ankle_pitch_joint",
                "left_ankle_roll_joint",
            ],
            preserve_order=True,
        )
        self.right_leg_ids, _ = self.robot.find_joints(
            name_keys=[
                "right_hip_pitch_joint",
                "right_hip_roll_joint",
                "right_hip_yaw_joint",
                "right_knee_joint",
                "right_ankle_pitch_joint",
                "right_ankle_roll_joint",
            ],
            preserve_order=True,
        )
        self.left_arm_ids, _ = self.robot.find_joints(
            name_keys=[
                "left_shoulder_pitch_joint",
                "left_shoulder_roll_joint",
                "left_shoulder_yaw_joint",
                "left_elbow_joint",
                "left_wrist_roll_joint",
            ],
            preserve_order=True,
        )
        self.right_arm_ids, _ = self.robot.find_joints(
            name_keys=[
                "right_shoulder_pitch_joint",
                "right_shoulder_roll_joint",
                "right_shoulder_yaw_joint",
                "right_elbow_joint",
                "right_wrist_roll_joint",
            ],
            preserve_order=True,
        )
        
        # Find waist joint IDs (new for G1 25DOF)
        self.waist_ids, _ = self.robot.find_joints(
            name_keys=[
                "waist_yaw_joint",
                "waist_roll_joint",
                "waist_pitch_joint",
            ],
            preserve_order=True,
        )

        self.ankle_joint_ids, _ = self.robot.find_joints(
            name_keys=["left_ankle_pitch_joint", "right_ankle_pitch_joint",
                       "left_ankle_roll_joint", "right_ankle_roll_joint"],
            preserve_order=True,
        )

        # Observation scaling and noise configuration
        self.obs_scales = self.cfg.normalization.obs_scales
        self.add_noise = self.cfg.noise.add_noise

        # Episode tracking buffers
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.sim_step_counter = 0
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        # Action and feet metrics buffers
        self.action = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.avg_feet_force_per_step = torch.zeros(
            self.num_envs, len(self.feet_cfg.body_ids), dtype=torch.float, device=self.device, requires_grad=False
        )
        self.avg_feet_speed_per_step = torch.zeros(
            self.num_envs, len(self.feet_cfg.body_ids), dtype=torch.float, device=self.device, requires_grad=False
        )
        
        # Initialize observation buffers
        self.init_obs_buffer()

    def visualize_motion(self, time):
        """Update the robot simulation state based on AMP motion capture data at a given time.

        This function sets the joint positions and velocities, root position and orientation,
        and linear/angular velocities according to the AMP motion frame at the specified time,
        then steps the simulation and updates the scene.

        Args:
            time (float): The time (in seconds) at which to fetch the AMP motion frame.
        """
        # visual_motion_frame = self.amp_loader_display.get_full_frame_at_time(0, time)
        visual_motion_frame = self._get_visual_frame_at_time(time)
        device = self.device
        
        # Debug: Print motion frame shape
        frame_dim = visual_motion_frame.shape[0]
        #print(f"[INFO] Motion data dimension: {frame_dim}")
        
        # Initialize DOF position and velocity tensors
        dof_pos = torch.zeros((self.num_envs, self.robot.num_joints), device=device)
        dof_vel = torch.zeros((self.num_envs, self.robot.num_joints), device=device)
        
        if frame_dim == 52:
            # 20DOF format (no waist joints)
            print("[INFO] Using 20DOF motion data format (no waist)")
            """
            20DOF format:
                [0:3]    - root position
                [3:6]    - root orientation (euler)
                [6:12]   - left leg (6)
                [12:18]  - right leg (6)
                [18:22]  - left arm (4: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow)
                [22:26]  - right arm (4)
                [26:29]  - root linear velocity
                [29:32]  - root angular velocity
                [32:38]  - left leg velocities
                [38:44]  - right leg velocities
                [44:48]  - left arm velocities
                [48:52]  - right arm velocities
            """
            # Set joint positions (leave waist and wrist at zero)
            dof_pos[:, self.left_leg_ids] = visual_motion_frame[6:12]
            dof_pos[:, self.right_leg_ids] = visual_motion_frame[12:18]
            dof_pos[:, self.left_arm_ids[:4]] = visual_motion_frame[18:22]  # Only first 4 joints
            dof_pos[:, self.right_arm_ids[:4]] = visual_motion_frame[22:26]
            
            # Set joint velocities
            dof_vel[:, self.left_leg_ids] = visual_motion_frame[32:38]
            dof_vel[:, self.right_leg_ids] = visual_motion_frame[38:44]
            dof_vel[:, self.left_arm_ids[:4]] = visual_motion_frame[44:48]
            dof_vel[:, self.right_arm_ids[:4]] = visual_motion_frame[48:52]
            
            # Velocities for root
            lin_vel = visual_motion_frame[26:29].clone()
            ang_vel = visual_motion_frame[29:32].clone()
            
        elif frame_dim == 64:
            # 25DOF format (with waist joints)
            # print("[INFO] Using 25DOF motion data format (with waist)")
            """
            25DOF format:
                [0:3]    - root position
                [3:6]    - root orientation (euler)
                [6:12]   - left leg (6)
                [12:18]  - right leg (6)  
                [18:21]  - waist (3: yaw, roll, pitch)
                [21:26]  - left arm (5: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll)
                [26:31]  - right arm (5)
                [31:34]  - root linear velocity
                [34:37]  - root angular velocity
                [37:43]  - left leg velocities
                [43:49]  - right leg velocities
                [49:52]  - waist velocities
                [52:57]  - left arm velocities
                [57:62]  - right arm velocities
                [62:63]  - left foot height
                [63:64]  - right foot height
            """
            # Set joint positions
            dof_pos[:, self.left_leg_ids] = visual_motion_frame[6:12]
            dof_pos[:, self.right_leg_ids] = visual_motion_frame[12:18]
            dof_pos[:, self.waist_ids] = visual_motion_frame[18:21]
            dof_pos[:, self.left_arm_ids] = visual_motion_frame[21:26]
            dof_pos[:, self.right_arm_ids] = visual_motion_frame[26:31]
            
            # Set joint velocities
            dof_vel[:, self.left_leg_ids] = visual_motion_frame[37:43]
            dof_vel[:, self.right_leg_ids] = visual_motion_frame[43:49]
            dof_vel[:, self.waist_ids] = visual_motion_frame[49:52]
            dof_vel[:, self.left_arm_ids] = visual_motion_frame[52:57]
            dof_vel[:, self.right_arm_ids] = visual_motion_frame[57:62]
            
            # Velocities for root
            lin_vel = visual_motion_frame[31:34].clone()
            ang_vel = visual_motion_frame[34:37].clone()
        else:
            raise ValueError(f"Unsupported motion frame dimension: {frame_dim}. Expected 52 (20DOF) or 64 (25DOF)")

        # Write joint states to simulation
        self.robot.write_joint_position_to_sim(dof_pos)
        self.robot.write_joint_velocity_to_sim(dof_vel)

        env_ids = torch.arange(self.num_envs, device=device)

        # Set root position
        root_pos = visual_motion_frame[:3].clone()

        # Convert euler angles to quaternion for root orientation
        euler = visual_motion_frame[3:6].cpu().numpy()
        quat_xyzw = Rotation.from_euler("XYZ", euler, degrees=False).as_quat()  # [x, y, z, w]
        quat_wxyz = torch.tensor(
            [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=torch.float32, device=device
        )
        
        # Note: lin_vel and ang_vel are already set in the format-specific block above

        # Construct root state: [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
        root_state = torch.zeros((self.num_envs, 13), device=device)
        root_state[:, 0:3] = torch.tile(root_pos.unsqueeze(0), (self.num_envs, 1))
        root_state[:, 3:7] = torch.tile(quat_wxyz.unsqueeze(0), (self.num_envs, 1))
        root_state[:, 7:10] = torch.tile(lin_vel.unsqueeze(0), (self.num_envs, 1))
        root_state[:, 10:13] = torch.tile(ang_vel.unsqueeze(0), (self.num_envs, 1))

        # Write root state and step simulation
        self.robot.write_root_state_to_sim(root_state, env_ids)
        self.sim.render()
        self.sim.step()
        self.scene.update(dt=self.step_dt)

    def compute_current_observations(self):
        """Compute current actor and critic observations from robot state.
        
        Returns:
            tuple: (current_actor_obs, current_critic_obs) - Observation tensors for actor and critic
        """
        robot = self.robot
        net_contact_forces = self.contact_sensor.data.net_forces_w_history

        # Extract robot state information
        ang_vel = robot.data.root_ang_vel_b
        projected_gravity = robot.data.projected_gravity_b
        command = self.command_generator.command
        joint_pos = robot.data.joint_pos - robot.data.default_joint_pos
        joint_vel = robot.data.joint_vel - robot.data.default_joint_vel
        action = self.action_buffer._circular_buffer.buffer[:, -1, :]
        root_lin_vel = robot.data.root_lin_vel_b
        
        # Determine feet contact state
        feet_contact = torch.max(torch.norm(net_contact_forces[:, :, self.feet_cfg.body_ids], dim=-1), dim=1)[0] > 0.5

        # Construct actor observations (proprioceptive + commands + history)
        current_actor_obs = torch.cat(
            [
                root_lin_vel * self.obs_scales.lin_vel,  # 3
                ang_vel * self.obs_scales.ang_vel,  # 3
                projected_gravity * self.obs_scales.projected_gravity,  # 3
                command * self.obs_scales.commands,  # 3
                joint_pos * self.obs_scales.joint_pos,  # 25 (DOF count for G1_25)
                joint_vel * self.obs_scales.joint_vel,  # 25
                action * self.obs_scales.actions,  # 25
            ],
            dim=-1,
        )

        # Critic observations include feet contact information
        current_critic_obs = torch.cat([current_actor_obs, feet_contact], dim=-1)

        return current_actor_obs, current_critic_obs

    def compute_observations(self):
        """Compute full observations with history, noise, and optional sensor data.
        
        Returns:
            tuple: (actor_obs, critic_obs) - Full observation tensors with history
        """
        # Get current observations
        current_actor_obs, current_critic_obs = self.compute_current_observations()
        
        # Add observation noise if enabled
        if self.add_noise:
            current_actor_obs += (2 * torch.rand_like(current_actor_obs) - 1) * self.noise_scale_vec

        # Update observation buffers with current observations
        self.actor_obs_buffer.append(current_actor_obs)
        self.critic_obs_buffer.append(current_critic_obs)

        # Flatten observation history
        actor_obs = self.actor_obs_buffer.buffer.reshape(self.num_envs, -1)
        critic_obs = self.critic_obs_buffer.buffer.reshape(self.num_envs, -1)
        
        # Add height scan data if enabled
        if self.cfg.scene.height_scanner.enable_height_scan:
            height_scan = (
                self.height_scanner.data.pos_w[:, 2].unsqueeze(1)
                - self.height_scanner.data.ray_hits_w[..., 2]
                - self.cfg.normalization.height_scan_offset
            ) * self.obs_scales.height_scan
            critic_obs = torch.cat([critic_obs, height_scan], dim=-1)
            if self.add_noise:
                height_scan += (2 * torch.rand_like(height_scan) - 1) * self.height_scan_noise_vec
            actor_obs = torch.cat([actor_obs, height_scan], dim=-1)

        # Add depth camera data if enabled
        if self.cfg.scene.depth_camera.enable_depth_camera:
            depth_image = self.depth_camera.data.output["distance_to_image_plane"]

            # Flatten depth image: (num_envs, height, width, 1) --> (num_envs, height * width)
            flattened_depth = depth_image.view(self.num_envs, -1)

            # Append the flattened depth data to observations
            actor_obs = torch.cat([actor_obs, flattened_depth], dim=-1)
            critic_obs = torch.cat([critic_obs, flattened_depth], dim=-1)

        # Clip observations to prevent extreme values
        actor_obs = torch.clip(actor_obs, -self.clip_obs, self.clip_obs)
        critic_obs = torch.clip(critic_obs, -self.clip_obs, self.clip_obs)

        return actor_obs, critic_obs

    def reset(self, env_ids):
        """Reset specified environments to initial state.
        
        Args:
            env_ids: Tensor of environment indices to reset
        """
        if len(env_ids) == 0:
            return

        # Reset feet metrics buffers
        self.avg_feet_force_per_step[env_ids] = 0.0
        self.avg_feet_speed_per_step[env_ids] = 0.0

        # Prepare extras dictionary for logging
        self.extras["log"] = dict()
        
        # Update terrain levels if curriculum is enabled
        if self.cfg.scene.terrain_generator is not None:
            if self.cfg.scene.terrain_generator.curriculum:
                terrain_levels = self.update_terrain_levels(env_ids)
                self.extras["log"].update(terrain_levels)

        # Reset scene and apply reset events
        self.scene.reset(env_ids)
        if "reset" in self.event_manager.available_modes:
            self.event_manager.apply(
                mode="reset",
                env_ids=env_ids,
                dt=self.step_dt,
                global_env_step_count=self.sim_step_counter // self.cfg.sim.decimation,
            )

        # Reset reward manager and update logs
        reward_extras = self.reward_manager.reset(env_ids)
        self.extras["log"].update(reward_extras)
        self.extras["time_outs"] = self.time_out_buf

        # Reset command generator and observation buffers
        self.command_generator.reset(env_ids)
        self.actor_obs_buffer.reset(env_ids)
        self.critic_obs_buffer.reset(env_ids)
        self.action_buffer.reset(env_ids)
        self.episode_length_buf[env_ids] = 0

        # Write reset state to simulation
        self.scene.write_data_to_sim()
        self.sim.forward()

    def step(self, actions: torch.Tensor):
        """Execute one step of the environment with the given actions.
        
        Args:
            actions: Action tensor from the policy
            
        Returns:
            tuple: (actor_obs, reward_buf, reset_buf, extras) - Step results
        """
        # Apply action delay and clip actions
        delayed_actions = self.action_buffer.compute(actions)
        self.action = torch.clip(delayed_actions, -self.clip_actions, self.clip_actions).to(self.device)

        # Convert normalized actions to joint positions
        processed_actions = self.action * self.action_scale + self.robot.data.default_joint_pos

        # Reset feet metrics for this step
        self.avg_feet_force_per_step = torch.zeros(
            self.num_envs, len(self.feet_cfg.body_ids), dtype=torch.float, device=self.device, requires_grad=False
        )
        self.avg_feet_speed_per_step = torch.zeros(
            self.num_envs, len(self.feet_cfg.body_ids), dtype=torch.float, device=self.device, requires_grad=False
        )
        
        # Execute decimation steps in physics simulation
        for _ in range(self.cfg.sim.decimation):
            self.sim_step_counter += 1
            self.robot.set_joint_position_target(processed_actions)
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)

            # Accumulate feet force and speed metrics
            self.avg_feet_force_per_step += torch.norm(
                self.contact_sensor.data.net_forces_w[:, self.feet_cfg.body_ids, :3], dim=-1
            )
            self.avg_feet_speed_per_step += torch.norm(self.robot.data.body_lin_vel_w[:, self.feet_body_ids, :], dim=-1)

        # Average feet metrics over decimation steps
        self.avg_feet_force_per_step /= self.cfg.sim.decimation
        self.avg_feet_speed_per_step /= self.cfg.sim.decimation

        # Render if not in headless mode
        if not self.headless:
            self.sim.render()

        # Update episode length counter
        self.episode_length_buf += 1

        # Update commands and apply interval events
        self.command_generator.compute(self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        # Check for resets and compute rewards
        self.reset_buf, self.time_out_buf = self.check_reset()
        reward_buf = self.reward_manager.compute(self.step_dt)
        self.reset_env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset(self.reset_env_ids)

        # Compute observations for next step
        actor_obs, critic_obs = self.compute_observations()
        self.extras["observations"] = {"critic": critic_obs}

        return actor_obs, reward_buf, self.reset_buf, self.extras

    def check_reset(self):
        """Check if environments should be reset due to termination conditions.
        
        Returns:
            tuple: (reset_buf, time_out_buf) - Boolean tensors indicating resets
        """
        net_contact_forces = self.contact_sensor.data.net_forces_w_history

        # Check for undesired contacts (termination condition)
        reset_buf = torch.any(
            torch.max(
                torch.norm(
                    net_contact_forces[:, :, self.termination_contact_cfg.body_ids],
                    dim=-1,
                ),
                dim=1,
            )[0]
            > 1.0,
            dim=1,
        )
        
        # Check for episode timeout
        time_out_buf = self.episode_length_buf >= self.max_episode_length
        reset_buf |= time_out_buf
        return reset_buf, time_out_buf

    def init_obs_buffer(self):
        """Initialize observation buffers and noise vectors."""
        # Setup observation noise if enabled
        if self.add_noise:
            actor_obs, _ = self.compute_current_observations()
            noise_vec = torch.zeros_like(actor_obs[0])
            noise_scales = self.cfg.noise.noise_scales
            
            # Configure noise for each observation component
            noise_vec[:3] = noise_scales.lin_vel * self.obs_scales.lin_vel
            noise_vec[3:6] = noise_scales.ang_vel * self.obs_scales.ang_vel
            noise_vec[6:9] = noise_scales.projected_gravity * self.obs_scales.projected_gravity
            noise_vec[9:12] = 0  # No noise on commands
            noise_vec[12: 12 + self.num_actions] = noise_scales.joint_pos * self.obs_scales.joint_pos
            noise_vec[12 + self.num_actions: 12 + self.num_actions * 2] = (
                noise_scales.joint_vel * self.obs_scales.joint_vel
            )
            noise_vec[12 + self.num_actions * 2: 12 + self.num_actions * 3] = 0.0  # No noise on actions
            self.noise_scale_vec = noise_vec

            # Setup height scan noise if enabled
            if self.cfg.scene.height_scanner.enable_height_scan:
                height_scan = (
                    self.height_scanner.data.pos_w[:, 2].unsqueeze(1)
                    - self.height_scanner.data.ray_hits_w[..., 2]
                    - self.cfg.normalization.height_scan_offset
                )
                height_scan_noise_vec = torch.zeros_like(height_scan[0])
                height_scan_noise_vec[:] = noise_scales.height_scan * self.obs_scales.height_scan
                self.height_scan_noise_vec = height_scan_noise_vec

        # Create circular buffers for observation history
        self.actor_obs_buffer = CircularBuffer(
            max_len=self.cfg.robot.actor_obs_history_length, batch_size=self.num_envs, device=self.device
        )
        self.critic_obs_buffer = CircularBuffer(
            max_len=self.cfg.robot.critic_obs_history_length, batch_size=self.num_envs, device=self.device
        )

    def update_terrain_levels(self, env_ids):
        """Update terrain difficulty levels based on robot progress (curriculum learning).
        
        Args:
            env_ids: Environment indices to update terrain for
            
        Returns:
            dict: Dictionary with terrain level metrics for logging
        """
        # Calculate distance traveled from spawn point
        distance = torch.norm(self.robot.data.root_pos_w[env_ids, :2] - self.scene.env_origins[env_ids, :2], dim=1)
        
        # Move to harder terrain if traveled far enough
        move_up = distance > self.scene.terrain.cfg.terrain_generator.size[0] / 2
        
        # Move to easier terrain if not making progress
        move_down = (
            distance < torch.norm(self.command_generator.command[env_ids, :2], dim=1) * self.max_episode_length_s * 0.5
        )
        move_down *= ~move_up
        
        # Update terrain origins
        self.scene.terrain.update_env_origins(env_ids, move_up, move_down)
        
        # Return logging metrics
        extras = {}
        extras["Curriculum/terrain_levels"] = torch.mean(self.scene.terrain.terrain_levels.float())
        return extras

    def get_observations(self):
        """Get current observations without stepping the simulation.
        
        Returns:
            tuple: (actor_obs, extras) - Current observations and extras dict
        """
        actor_obs, critic_obs = self.compute_observations()
        self.extras["observations"] = {"critic": critic_obs}
        return actor_obs, self.extras

    def get_amp_obs_for_expert_trans(self):
        """Get AMP (Adversarial Motion Priors) observations for expert trajectories.
        
        提取机器人状态，匹配GMR数据格式（25DOF机器人为64维）：
        [root_pos(3), root_euler(3), dof_pos(25), root_lin_vel(3), 
         root_ang_vel(3), dof_vel(25), left_foot_height(1), right_foot_height(1)]
        
        Returns:
            torch.Tensor: AMP observations tensor (num_envs, 64)
        """
        # 1. 根位置（世界坐标系）- 3维
        # Root position in world frame [ z]
        root_pos = self.robot.data.root_pos_w[:, 2]
        
        # 2. 根欧拉角（XYZ顺序，内旋）- 3维
        # Convert root quaternion [w,x,y,z] to Euler angles [roll, pitch, yaw]
        root_quat_w = self.robot.data.root_quat_w  # (num_envs, 4) [w,x,y,z]
        root_euler = self._quat_to_euler_xyz(root_quat_w)  # (num_envs, 3)
        
        # 3. 所有关节角度（按机器人定义顺序）- 25维
        # All joint positions (25 DOF) in robot's joint order
        dof_pos = self.robot.data.joint_pos  # (num_envs, 25)
        
        # 4. 根节点线速度（世界坐标系）- 3维
        # Root linear velocity in world frame [vx, vy, vz]
        root_lin_vel = self.robot.data.root_lin_vel_w[:, :3]
        
        # 5. 根节点角速度（世界坐标系）- 3维
        # Root angular velocity in world frame [wx, wy, wz]
        root_ang_vel = self.robot.data.root_ang_vel_w[:, :3]
        
        # 6. 所有关节速度 - 25维
        # All joint velocities (25 DOF)
        dof_vel = self.robot.data.joint_vel  # (num_envs, 25)
        
        # 7. 左脚高度（相对于地面）- 1维
        # Left foot height above ground level
        left_foot_z = self.robot.data.body_pos_w[:, self.feet_body_ids[0], 2]
        # 假设地面高度为环境原点的Z坐标
        # Assume ground height is the Z coordinate of environment origin
        ground_height = self.scene.env_origins[:, 2]
        left_foot_height = (left_foot_z - ground_height).unsqueeze(-1)
        
        # 8. 右脚高度（相对于地面）- 1维
        # Right foot height above ground level
        right_foot_z = self.robot.data.body_pos_w[:, self.feet_body_ids[1], 2]
        right_foot_height = (right_foot_z - ground_height).unsqueeze(-1)
        
        # 拼接所有AMP观测（总计64维）
        # Concatenate all AMP observations (total 64 dims for 25DOF robot)
        amp_obs = torch.cat(
            [
                root_pos,              # 3 - 根位置
                root_euler,            # 3 - 根欧拉角
                dof_pos,               # 25 - 关节角度
                root_lin_vel,          # 3 - 根线速度
                root_ang_vel,          # 3 - 根角速度
                dof_vel,               # 25 - 关节速度
                left_foot_height,      # 1 - 左脚高度
                right_foot_height,     # 1 - 右脚高度
            ],
            dim=-1,
        )
        
        return amp_obs  # (num_envs, 64)
    
    def _quat_to_euler_xyz(self, quat):
        """将四元数转换为欧拉角（XYZ顺序，内旋）
        Convert quaternion to Euler angles (XYZ order, intrinsic rotations).
        
        Args:
            quat: (num_envs, 4) quaternion in [w, x, y, z] format
            
        Returns:
            euler: (num_envs, 3) Euler angles [roll, pitch, yaw] in radians
        """
        # 提取四元数分量 Extract quaternion components
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        
        # Roll (x-axis rotation) 绕X轴旋转
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = torch.atan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation) 绕Y轴旋转
        sinp = 2 * (w * y - z * x)
        # 限制在[-1, 1]范围内避免asin域错误
        # Clamp to [-1, 1] to avoid asin domain errors
        sinp = torch.clamp(sinp, -1.0, 1.0)
        pitch = torch.asin(sinp)
        
        # Yaw (z-axis rotation) 绕Z轴旋转
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = torch.atan2(siny_cosp, cosy_cosp)
        
        # 拼接为 [roll, pitch, yaw]
        # Stack as [roll, pitch, yaw]
        euler = torch.stack([roll, pitch, yaw], dim=-1)
        
        return euler

    @staticmethod
    def seed(seed: int = -1) -> int:
        """Set random seed for reproducibility.
        
        Args:
            seed: Random seed value
            
        Returns:
            int: The seed that was set
        """
        try:
            import omni.replicator.core as rep  # type: ignore

            rep.set_global_seed(seed)
        except ModuleNotFoundError:
            pass
        return torch_utils.set_seed(seed)
