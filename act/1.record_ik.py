import mujoco.viewer
import mujoco
import time
import ikpy.chain
import transforms3d as tf
import numpy as np
import random
import os
import h5py
import pickle
from contextlib import nullcontext  # <--- 新增工具：空上下文管理器

# ================= 配置 =================
DATA_DIR = "data_act"
NUM_EPISODES = 100
CONTROL_DT = 0.02           # 50Hz
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
CAMERA_NAMES = ['top', 'wrist']
USE_VIEWER = False  # 【开关】True=看画面(调试用), False=极速后台采集

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# ================= 辅助类 (保持不变) =================
class DataCollector:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.reset_buffer()
        self.renderer = None

    def reset_buffer(self):
        self.buffer = {
            'qpos': [], 'qvel': [], 'action': [],
            'images': {name: [] for name in CAMERA_NAMES}
        }

    def init_renderer(self, model, width, height):
        self.renderer = mujoco.Renderer(model, height=height, width=width)

    def capture_images(self, data):
        name_map = {'top': 'top_camera', 'wrist': 'wrist_camera'}
        imgs = {}
        for short_name, xml_name in name_map.items():
            self.renderer.update_scene(data, camera=xml_name)
            imgs[short_name] = self.renderer.render()
        return imgs

    def add_frame(self, qpos, qvel, action, images):
        self.buffer['qpos'].append(qpos)
        self.buffer['qvel'].append(qvel)
        self.buffer['action'].append(action)
        for name in CAMERA_NAMES:
            self.buffer['images'][name].append(images[name])

    def save_episode(self, episode_idx):
        data_len = len(self.buffer['qpos'])
        if data_len == 0: return
        
        qpos = np.array(self.buffer['qpos'], dtype=np.float32)
        qvel = np.array(self.buffer['qvel'], dtype=np.float32)
        action = np.array(self.buffer['action'], dtype=np.float32)
        
        filename = os.path.join(self.save_dir, f"episode_{episode_idx}.hdf5")
        with h5py.File(filename, 'w') as root:
            root.attrs['sim'] = True
            root.attrs['num_samples'] = data_len
            root.create_dataset('action', data=action)
            obs = root.create_group('observations')
            obs.create_dataset('qpos', data=qpos)
            obs.create_dataset('qvel', data=qvel)
            image_grp = obs.create_group('images')
            for name in CAMERA_NAMES:
                image_grp.create_dataset(name, data=np.array(self.buffer['images'][name], dtype=np.uint8))
        
        print(f"💾 Saved Episode {episode_idx} ({data_len} steps)")
        self.reset_buffer()

class JointSpaceTrajectory:
    def __init__(self, start_joints, end_joints, steps):
        self.start_joints = np.array(start_joints)
        self.end_joints = np.array(end_joints)
        self.steps = steps
        self.current_step = 0
        self.step_vec = (self.end_joints - self.start_joints) / self.steps
        self.finished = False

    def get_next_waypoint(self):
        if self.current_step < self.steps:
            self.current_step += 1
            return self.start_joints + self.step_vec * self.current_step
        else:
            self.finished = True
            return self.end_joints

def get_ik_solution(chain, target_pos, target_euler_deg, initial_guess):
    target_euler_rad = np.radians(target_euler_deg)
    target_orientation = tf.euler.euler2mat(*target_euler_rad)
    try:
        return chain.inverse_kinematics(target_position=target_pos, target_orientation=target_orientation, orientation_mode="all", initial_position=initial_guess)
    except: return None

def reset_env(model, data):
    mujoco.mj_resetData(model, data)
    home_pose = [0, -0.785, 0, -2.356, 0, 1.571, 0.785] 
    data.qpos[:7] = home_pose; data.qpos[7] = 0.04; data.qpos[8] = 0.04 
    data.ctrl[:7] = home_pose; data.ctrl[7] = 255 
    cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube")
    # 调试开始
    cube_joint_name = "cube" # 你的代码里写的是这个名字
    cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, cube_joint_name)
    
    print(f"DEBUG: 查找关节 '{cube_joint_name}', ID结果: {cube_id}") # <--- 加上这句

    if cube_id != -1:
        adr = model.jnt_qposadr[cube_id]
        rx = random.uniform(0.45, 0.65)
        ry = random.uniform(-0.2, 0.2)
        print(f"DEBUG: 生成随机位置: ({rx:.4f}, {ry:.4f})") # <--- 加上这句
        data.qpos[adr] = rx
        data.qpos[adr+1] = ry
        data.qpos[adr+2] = 0.03
        data.qpos[adr+3:adr+7] = [1, 0, 0, 0]
    else:
        print("❌ 错误：没找到方块的关节！随机化被跳过，方块将保持在 XML 默认位置。") # <--- 加上这句

    for _ in range(100): mujoco.mj_step(model, data)
    return True

# ================= 主程序 (优化了 Viewer 逻辑) =================

def main():
    model = mujoco.MjModel.from_xml_path('franka_emika_panda/scene.xml')
    data = mujoco.MjData(model)
    sim_dt = model.opt.timestep 
    n_substeps = int(CONTROL_DT / sim_dt)
    
    gripper_id = -1
    for i in range(model.nu):
        if 'finger' in mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i):
            gripper_id = i; break

    collector = DataCollector(DATA_DIR)
    collector.init_renderer(model, IMAGE_WIDTH, IMAGE_HEIGHT)

    my_chain = ikpy.chain.Chain.from_urdf_file("franka_emika_panda/panda.urdf", base_elements=["panda_link0"], last_link_vector=[0, 0, 0.107], active_links_mask=[False, True, True, True, True, True, True, True, False, False])

    STATE_INIT, STATE_APPROACH, STATE_GRASP, STATE_LIFT = 0, 1, 2, 3
    current_state = STATE_INIT
    episode_count = 0
    traj = None
    q_lift = None
    grasp_timer = 0
    
    # --- 关键修改：上下文管理器逻辑 ---
    if USE_VIEWER:
        print("👁️ 可视化模式：启动窗口...")
        viewer_ctx = mujoco.viewer.launch_passive(model, data)
    else:
        print("🚀 Headless 模式：极速后台采集...")
        viewer_ctx = nullcontext() # 创建一个空的上下文，什么都不做，但允许使用 'with'

    # 使用 with 语句统一管理，无论是否有 viewer
    with viewer_ctx as viewer:
        # 如果有 viewer，初始化视角
        if viewer:
            viewer.cam.lookat[:] = [0.5, 0, 0.2]
            viewer.cam.distance = 1.5; viewer.cam.azimuth = 130; viewer.cam.elevation = -30
        
        while episode_count < NUM_EPISODES:
            # 如果是可视化模式，且窗口被关闭，则退出
            if viewer and not viewer.is_running():
                print("⚠️ 窗口已关闭，停止采集")
                break
            
            # --- 状态机逻辑 ---
            if current_state == STATE_INIT:
                print(f"Generating Episode {episode_count}...")
                reset_env(model, data)
                cube_pos = data.body("cube").xpos
                grasp_target = cube_pos.copy(); grasp_target[2] = 0.015 
                grasp_euler = [180, 0, 45]
                guess = [0.0]*10; guess[4] = -1.57; guess[6] = 1.57
                ik_grasp = get_ik_solution(my_chain, grasp_target, grasp_euler, guess)
                if ik_grasp is None: continue 
                q_grasp = ik_grasp[1:8]
                lift_target = cube_pos + [0, 0, 0.2]
                ik_lift = get_ik_solution(my_chain, lift_target, grasp_euler, ik_grasp)
                if ik_lift is None: continue
                q_lift = ik_lift[1:8]
                curr_q = data.qpos[:7].copy()
                traj = JointSpaceTrajectory(curr_q, q_grasp, steps=100)
                current_state = STATE_APPROACH
            
            target_q = data.ctrl[:7].copy()
            if traj and not traj.finished:
                target_q = traj.get_next_waypoint()
                data.ctrl[:7] = target_q

            if current_state == STATE_APPROACH:
                data.ctrl[gripper_id] = 255
                if traj.finished: current_state = STATE_GRASP; grasp_timer = 0
            elif current_state == STATE_GRASP:
                data.ctrl[gripper_id] = 0
                grasp_timer += 1
                if grasp_timer > 30: current_state = STATE_LIFT; curr_q = data.qpos[:7].copy(); traj = JointSpaceTrajectory(curr_q, q_lift, steps=100)
            elif current_state == STATE_LIFT:
                data.ctrl[gripper_id] = 0
                if traj.finished:
                    collector.save_episode(episode_count)
                    episode_count += 1
                    current_state = STATE_INIT
                    continue

            # --- 物理步进 ---
            for _ in range(n_substeps):
                mujoco.mj_step(model, data)
            
            # --- 仅在可视化模式下同步画面 ---
            if viewer:
                viewer.sync()
            
            # --- 数据采集 (始终执行) ---
            if current_state != STATE_INIT:
                imgs = collector.capture_images(data) # 这一步在 Headless 模式下依然会渲染相机，只是不显示到屏幕
                qpos = data.qpos[:9].copy()
                qvel = data.qvel[:9].copy()
                action = data.ctrl[:8].copy()
                collector.add_frame(qpos, qvel, action, imgs)

    print("采集结束！")

if __name__ == "__main__":
    main()