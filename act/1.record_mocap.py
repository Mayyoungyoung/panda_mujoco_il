import mujoco
import mujoco.viewer
import numpy as np
import time
import pickle
import os

# --- 配置 ---
XML_PATH = "./franka_emika_panda/scene_mocap.xml"
SAVE_DIR = "data"
NUM_EPISODES = 10  # 打算录多少集
MAX_STEPS = 400    # 每集最大步数 (防止死循环)

class AutoCollector:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(XML_PATH)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        
        # 确保保存目录存在
        os.makedirs(SAVE_DIR, exist_ok=True)

        # 缓存 ID
        self.hand_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self.mocap_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "mocap_hand")
        self.cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self.mocap_id = self.model.body_mocapid[self.mocap_body_id]
        
        # 查找夹爪执行器 (假设是最后一个或包含 gripper 名字)
        self.gripper_actuator_id = -1
        for i in range(self.model.nu):
            if 'finger' in mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i):
                self.gripper_actuator_id = i
                break

    def get_pixel_obs(self):
        images = {}
        for cam in ["top_camera", "wrist_camera"]:
            self.renderer.update_scene(self.data, camera=cam)
            images[cam] = self.renderer.render()
        return images

    def reset_env(self):
        """重置环境，随机放置方块"""
        mujoco.mj_resetData(self.model, self.data)
        
        # 1. 随机方块位置 (X: 0.4~0.6, Y: -0.2~0.2)
        random_x = np.random.uniform(0.4, 0.6)
        random_y = np.random.uniform(-0.2, 0.2)
        
        # 设置方块的 qpos (7维: 3位置 + 4四元数)
        # 注意：freejoint 的 qpos 索引通常在最后，或者用 joint 名字查
        cube_jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube") # 假设 joint 名字也是 cube
        if cube_jnt_id == -1:
             # 如果是用 body name 查 qpos 地址:
             cube_qpos_adr = self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")] 
             # 这里简化处理，假设只有一个 free joint
             pass

        # 简单暴力的重置方法：直接改 xpos 没用，要改 qpos
        # 找到 cube 对应的 qpos 起始位置
        # 假设 cube 是第 0 个 free joint
        qpos_adr = self.model.jnt_qposadr[0] # 这里需要根据你的 xml 实际情况调整
        
        self.data.qpos[qpos_adr] = random_x
        self.data.qpos[qpos_adr+1] = random_y
        self.data.qpos[qpos_adr+2] = 0.025 # Z 高度 (半个边长)
        
        # 2. 初始化 Mocap 到机械臂初始位置 (避免爆炸)
        mujoco.mj_forward(self.model, self.data)
        self.data.mocap_pos[self.mocap_id] = self.data.xpos[self.hand_body_id]
        self.data.mocap_quat[self.mocap_id] = self.data.xquat[self.hand_body_id]
        
        mujoco.mj_forward(self.model, self.data)
        return [], [] # 清空数据缓存

    def move_mocap_smooth(self, target_pos, target_quat, steps=50, gripper_open=True, record_list=None):
        """平滑移动 Mocap，并在过程中录制数据"""
        start_pos = self.data.mocap_pos[self.mocap_id].copy()
        start_quat = self.data.mocap_quat[self.mocap_id].copy()
        
        # 生成插值路径
        for i in range(steps):
            alpha = (i + 1) / steps
            
            # 线性插值位置
            current_target = (1 - alpha) * start_pos + alpha * target_pos
            self.data.mocap_pos[self.mocap_id] = current_target
            
            # (可选) 四元数插值 slerp，这里简化为保持不变或直接设置
            if target_quat is not None:
                self.data.mocap_quat[self.mocap_id] = target_quat

            # 控制夹爪
            ctrl_val = 255 if gripper_open else 0
            if self.gripper_actuator_id != -1:
                self.data.ctrl[self.gripper_actuator_id] = ctrl_val
            
            # 物理步进
            mujoco.mj_step(self.model, self.data)
            
            # 录制数据
            if record_list is not None:
                obs = {
                    'qpos': self.data.qpos.copy(),
                    'qvel': self.data.qvel.copy(),
                    'images': self.get_pixel_obs(),
                    'ctrl': self.data.ctrl.copy(),
                    'mocap_pose': np.concatenate([current_target, self.data.mocap_quat[self.mocap_id]])
                }
                record_list.append(obs)

    def collect(self):
        # 使用 headless 模式或者是被动 viewer
        # 这里用 passive viewer 方便你看过程，正式跑可以去掉
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            
            for episode_idx in range(NUM_EPISODES):
                print(f"🎬 Episode {episode_idx+1}/{NUM_EPISODES}")
                episode_data = []
                
                # 1. 重置
                self.reset_env()
                
                # 获取方块真实位置
                cube_pos = self.data.xpos[self.cube_body_id].copy()
                
                # 定义关键点
                # Point A: 准备姿势 (高处)
                home_pos = np.array([0.3, 0, 0.5])
                home_quat = np.array([0, 1, 0, 0]) # 抓握姿态 (需根据你的机械臂调整)
                
                # Point B: 方块正上方 (Hover)
                hover_pos = cube_pos.copy()
                hover_pos[2] += 0.2
                
                # Point C: 抓取位 (Grasp)
                grasp_pos = cube_pos.copy()
                grasp_pos[2] -= 0.01 # 稍微陷进去一点保证接触
                
                # --- 状态机执行 ---
                
                # Phase 1: 移动到方块上方
                self.move_mocap_smooth(hover_pos, home_quat, steps=60, gripper_open=True, record_list=episode_data)
                
                # Phase 2: 下降
                self.move_mocap_smooth(grasp_pos, home_quat, steps=40, gripper_open=True, record_list=episode_data)
                
                # Phase 3: 闭合夹爪 (位置不动，只动夹爪，多给点时间让物理稳定)
                for _ in range(20):
                    self.move_mocap_smooth(grasp_pos, home_quat, steps=1, gripper_open=False, record_list=episode_data)
                
                # Phase 4: 抬起
                lift_target = grasp_pos.copy()
                lift_target[2] += 0.3
                self.move_mocap_smooth(lift_target, home_quat, steps=60, gripper_open=False, record_list=episode_data)
                
                # 刷新画面
                viewer.sync()
                
                # --- 保存 ---
                save_path = os.path.join(SAVE_DIR, f"episode_{episode_idx}.pkl")
                with open(save_path, 'wb') as f:
                    pickle.dump(episode_data, f)
                print(f"   ✅ Saved {len(episode_data)} frames.")

if __name__ == "__main__":
    collector = AutoCollector()
    collector.collect()