# replay_motion_fixed.py
from isaacgym import gymapi, gymtorch
import os
import numpy as np
import torch
from time import sleep

# =============================
# 配置路径
# =============================
ROBOT_URDF = "/home/wan/Documents/ResMimic/assets/g1/g1_custom_collision_29dof.urdf"
OBJECT_URDF = "/home/wan/Documents/ResMimic/legged_gym/assets/www_box/www_box.urdf"
MOTION_FILE = "/home/wan/Documents/ResMimic/legged_gym/assets/motions/1_original.npz"

# =============================
# 初始化 Isaac Gym
# =============================
gym = gymapi.acquire_gym()
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.dt = 1/60.0
sim_params.use_gpu_pipeline = False
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise RuntimeError("Failed to create viewer")

# =============================
# 创建环境
# =============================
num_envs = 1
spacing = 2.0
envs = []
for i in range(num_envs):
    env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
    env_upper = gymapi.Vec3(spacing, spacing, spacing)
    env_handle = gym.create_env(sim, env_lower, env_upper, 1)
    envs.append(env_handle)

# =============================
# 加载 URDF
# =============================
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = False
asset_options.disable_gravity = True
asset_options.collapse_fixed_joints = True

robot_asset = gym.load_asset(sim, os.path.dirname(ROBOT_URDF), os.path.basename(ROBOT_URDF), asset_options)
object_asset = gym.load_asset(sim, os.path.dirname(OBJECT_URDF), os.path.basename(OBJECT_URDF), asset_options)

# =============================
# 创建 Actor
# =============================
start_pose = gymapi.Transform()
start_pose.p = gymapi.Vec3(0.0, 0.0, 0.5)
robot_handle = gym.create_actor(envs[0], robot_asset, start_pose, "robot", 0, 0, 0)

start_pose_obj = gymapi.Transform()
start_pose_obj.p = gymapi.Vec3(0.5, 0.0, 10.0)
object_handle = gym.create_actor(envs[0], object_asset, start_pose_obj, "object", 0, 0, 0)

num_robot_dofs = gym.get_asset_dof_count(robot_asset)

# =============================
# 获取 tensor
# =============================
actor_root_state = gym.acquire_actor_root_state_tensor(sim)
root_state_tensor = gymtorch.wrap_tensor(actor_root_state)

dof_state_tensor = gym.acquire_dof_state_tensor(sim)
dof_state_tensor = gymtorch.wrap_tensor(dof_state_tensor)

# Actor 索引 (ISAAC GYM 3.3: 用 env 索引数组)
env_ids_tensor = torch.tensor([0], dtype=torch.int32)
env_ids_tensor_native = gymtorch.unwrap_tensor(env_ids_tensor)

# =============================
# 加载 Motion 文件
# =============================
motion_data = np.load(MOTION_FILE, allow_pickle=True)
qpos_seq = motion_data['qpos']
qpos_seq = torch.tensor(qpos_seq, dtype=torch.float32)

# =============================
# 重放循环
# =============================
print("Starting replay...")

while not gym.query_viewer_has_closed(viewer):
    for frame_idx in range(qpos_seq.shape[0]):
        frame_qpos = qpos_seq[frame_idx]

        # robot base
        root_state_tensor[0, 0:3] = frame_qpos[0:3].clone()
        root_state_tensor[0, 6] = frame_qpos[3].clone()
        root_state_tensor[0, 3:6] = frame_qpos[4:7].clone()
        root_state_tensor[0, 7:13] = 0.0  # zero velocity

        # robot joints
        for i in range(num_robot_dofs):
            dof_state_tensor[i, 0] = frame_qpos[7 + i].clone()
            dof_state_tensor[i, 1] = 0.0

        # 从 frame_qpos 取出物体的四元数 wxyz
        q_wxyz = frame_qpos[num_robot_dofs + 3 : num_robot_dofs + 7]

        # 转换成 xyzw
        q_xyzw = torch.zeros(4, dtype=torch.float32)
        q_xyzw[0] = q_wxyz[1]  # x
        q_xyzw[1] = q_wxyz[2]  # y
        q_xyzw[2] = q_wxyz[3]  # z
        q_xyzw[3] = q_wxyz[0]  # w

        # object base
        root_state_tensor[1, 0:3] = frame_qpos[num_robot_dofs:3+num_robot_dofs].clone()
        root_state_tensor[1, 3:7] = q_xyzw.clone()
        root_state_tensor[1, 7:13] = 0.0

        # 更新仿真
        gym.set_actor_root_state_tensor_indexed(sim, gymtorch.unwrap_tensor(root_state_tensor), env_ids_tensor_native, len(env_ids_tensor))
        gym.set_dof_state_tensor_indexed(sim, gymtorch.unwrap_tensor(dof_state_tensor), env_ids_tensor_native, len(env_ids_tensor))

        # Step visualization
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim)
        sleep(sim_params.dt)

# =============================
# 清理
# =============================
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
print("Replay finished.")