import mujoco
import mujoco.viewer
import numpy as np
import cv2
import time
import pickle

# --- 配置 ---
XML_PATH = "./franka_emika_panda/scene.xml"
CAMERA_NAMES = ["top_camera", "wrist_camera"]
DT = 0.02  # 50Hz 采样率 (ACT 默认频率)

class DataRecorder:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(XML_PATH)
        self.data = mujoco.MjData(self.model)
        
        # 初始化渲染器
        self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        
        # 状态变量
        self.recording = False
        self.gripper_open = True
        self.episode_data = [] # 存储这一集的数据
        
        # 获取夹爪 actuator 的 ID (假设名字包含 'finger' 或 'gripper')
        # 如果你的 panda.xml actuator 名字不同，请修改这里
        self.gripper_actuator_id = -1
        for i in range(self.model.nu):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if 'finger' in name or 'gripper' in name:
                self.gripper_actuator_id = i
                print(f"Found Gripper Actuator: {name} (ID: {i})")
                break
        
        # 初始化 Mocap 位置到机械臂当前位置
        self.init_mocap()

    def init_mocap(self):
        """关键修复：初始化时让绿球吸附到机械臂手上，消除其实距离"""
        # 1. 先让物理引擎刷新一次，计算出机械臂当前的真实位置
        mujoco.mj_forward(self.model, self.data)
        
        # 2. 获取机械臂末端(hand)的 ID
        hand_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        
        # 3. 获取 Mocap Body (mocap_hand) 的 ID
        # 如果你现在只剩这一个mocap body，它的 ID 应该是 0，但为了保险我们用名字查
        mocap_name = "mocap_hand"
        mocap_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, mocap_name)
        
        if hand_id != -1 and mocap_body_id != -1:
            # 获取他在 mocap 数组里的索引
            mocap_id = self.model.body_mocapid[mocap_body_id]
            
            # === 核心操作 ===
            # 将绿球瞬间移动到机械臂手心的位置
            self.data.mocap_pos[mocap_id] = self.data.xpos[hand_id]
            # 必须！！！同时也复制旋转角度 (Quat)，否则手腕会疯狂扭曲
            self.data.mocap_quat[mocap_id] = self.data.xquat[hand_id]
            
            # 再次刷新，让物理引擎知道它们现在在一起了
            mujoco.mj_forward(self.model, self.data)
            print(f"✅ Mocap 已对齐到: {self.data.xpos[hand_id]}")
        else:
            print("❌ ID 查找失败，请检查 XML 中的 body 名字是否叫 'hand' 和 'mocap_hand'")

    def get_pixel_obs(self):
        """获取所有摄像头的图像"""
        images = {}
        for cam_name in CAMERA_NAMES:
            self.renderer.update_scene(self.data, camera=cam_name)
            images[cam_name] = self.renderer.render()
        return images

    def key_callback(self, keycode):
        """键盘回调"""
        # Space (32): 切换夹爪
        if keycode == 32: 
            self.gripper_open = not self.gripper_open
            print(f"Gripper: {'Open' if self.gripper_open else 'Close'}")
        
        # R (82): 开始/停止录制
        elif keycode == 82: 
            self.recording = not self.recording
            if self.recording:
                self.episode_data = [] # 清空旧数据
                print("🔴 开始录制... (操作机械臂抓取物体)")
            else:
                print("ww 暂停录制.")

        # S (83): 保存数据
        elif keycode == 83:
            self.save_data()

    def save_data(self):
        if not self.episode_data:
            print("数据为空，无法保存")
            return
        
        filename = f"episode_{int(time.time())}.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(self.episode_data, f)
        print(f"💾 数据已保存至 {filename} (帧数: {len(self.episode_data)})")
        self.episode_data = [] # 保存后清空

    def run(self):
        with mujoco.viewer.launch_passive(self.model, self.data, key_callback=self.key_callback) as viewer:
            print("========================================")
            print("操作指南:")
            print("1. 【双击】绿色球体选中它。")
            print("2. 按住 【Ctrl + 左键】 拖动平移。")
            print("3. 按住 【Ctrl + 右键】 拖动旋转。")
            print("4. 按 【Space】 键开合夹爪。")
            print("5. 准备好后，按 【R】 键开始录制。")
            print("6. 完成后，按 【S】 键保存数据。")
            print("========================================")

            last_time = time.time()
            
            while viewer.is_running():
                step_start = time.time()

                # 1. 应用夹爪控制
                # 假设夹爪控制范围是 0~255 (Panda通常是位置控制 0~0.04)
                # 这里给一个简化的控制逻辑，具体数值取决于你的 panda.xml actuator 配置
                ctrl_val = 255 if self.gripper_open else 0
                self.data.ctrl[7] = ctrl_val 

                # 2. 物理步进
                mujoco.mj_step(self.model, self.data)
                viewer.sync()

                # 3. 录制数据 (限制频率)
                if self.recording and (time.time() - last_time >= DT):
                    obs = {
                        'qpos': self.data.qpos.copy(), # 关节位置 (包含机械臂和夹爪)
                        'qvel': self.data.qvel.copy(), # 关节速度
                        'images': self.get_pixel_obs(), # 图像字典
                        'ctrl': self.data.ctrl.copy()  # 控制信号
                    }
                    self.episode_data.append(obs)
                    last_time = time.time()
                    print(f"\rRecording... Frames: {len(self.episode_data)}", end="")

# 运行
if __name__ == "__main__":
    recorder = DataRecorder()
    recorder.run()