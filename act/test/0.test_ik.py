import mujoco.viewer
import mujoco
import time
import ikpy.chain
import transforms3d as tf
import numpy as np

# ================= 辅助类与函数 =================

def viewer_init(viewer):
    """初始化视角，方便观察抓取"""
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    viewer.cam.lookat[:] = [0.5, 0, 0.2] # 看向桌子中心
    viewer.cam.distance = 1.5
    viewer.cam.azimuth = 130
    viewer.cam.elevation = -30

class JointSpaceTrajectory:
    """关节空间轨迹规划"""
    def __init__(self, start_joints, end_joints, steps):
        self.start_joints = np.array(start_joints)
        self.end_joints = np.array(end_joints)
        self.steps = steps
        self.current_step = 0
        self.step_vec = (self.end_joints - self.start_joints) / self.steps
        self.waypoint = self.start_joints.copy()
        self.finished = False

    def get_next_waypoint(self):
        if self.current_step < self.steps:
            self.waypoint += self.step_vec
            self.current_step += 1
        else:
            self.waypoint = self.end_joints
            self.finished = True
        return self.waypoint

def get_ik_solution(chain, target_pos, target_euler_deg, current_mask, initial_guess):
    """封装 IK 计算过程"""
    # 1. 角度转弧度
    target_euler_rad = np.radians(target_euler_deg)
    target_orientation = tf.euler.euler2mat(*target_euler_rad)
    
    # 2. 计算
    print(f"正在计算目标位置 {target_pos} 的 IK...")
    try:
        joint_angles = chain.inverse_kinematics(
            target_position=target_pos, 
            target_orientation=target_orientation, 
            orientation_mode="all", 
            initial_position=initial_guess
        )
        return joint_angles
    except ValueError as e:
        print(f"IK计算失败: {e}")
        return None

# ================= 主程序 =================

def main():
    # 1. 加载模型
    model = mujoco.MjModel.from_xml_path('franka_emika_panda/scene.xml')
    data = mujoco.MjData(model)

    # 2. 初始化环境 (Panda Home Pose)
    # 避免初始奇异点，并将夹爪设置为张开 (0.04)
    # qpos 结构: [J1...J7, Finger1, Finger2, CubeX...CubeQw...]
    home_pose = [0, -0.785, 0, -2.356, 0, 1.571, 0.785] 
    data.qpos[:7] = home_pose
    data.qpos[7] = 0.04 # 左指头张开
    data.qpos[8] = 0.04 # 右指头张开
    
    # 同步控制量
    data.ctrl[:7] = home_pose
    data.ctrl[7] = 255 # 夹爪张开
    
    mujoco.mj_step(model, data) # 刷新状态

    # 3. 初始化 IK 链
    # 【重要】加入 last_link_vector，让 IK 目标点直接变为指尖中心
    my_chain = ikpy.chain.Chain.from_urdf_file(
        "franka_emika_panda/panda.urdf",
        base_elements=["panda_link0"],
        last_link_vector=[0, 0, 0.107] 
    )
    
    # 链长变为 10 (0:Base ... 7:J7, 8:Flange, 9:Tip)
    active_mask = [False, True, True, True, True, True, True, True, False, False]
    my_chain.active_links_mask = active_mask

    # ------------------ 动作规划 ------------------

    # 动作 A: 抓取点 (正方体中心)
    # 注意：Z=0.02 是正方体中心高度。稍微向下压一点点 (0.015) 确保接触
    grasp_pos = [0.5, 0.0, 0.015] 
    # 姿态：垂直向下 (180度翻转)
    grasp_euler = [180, 0, 0]
    
    # 初始猜测 (关键：J4弯曲)
    ref_pos_guess = [0.0] * 10
    ref_pos_guess[4] = -1.57
    ref_pos_guess[6] = 1.57

    # 计算 A 的解
    ik_result_grasp = get_ik_solution(my_chain, grasp_pos, grasp_euler, active_mask, ref_pos_guess)
    if ik_result_grasp is None: return
    
    # 提取 7个关节角
    q_grasp = ik_result_grasp[1:8]

    # 动作 B: 抬起点 (Z轴抬高)
    lift_pos = [0.5, 0.0, 0.3] 
    lift_euler = [180, 0, 0]
    
    # 【技巧】用抓取姿态作为抬起姿态的初始猜测，保证动作连续性
    ref_pos_lift = ik_result_grasp.copy()
    
    # 计算 B 的解
    ik_result_lift = get_ik_solution(my_chain, lift_pos, lift_euler, active_mask, ref_pos_lift)
    if ik_result_lift is None: return
    
    q_lift = ik_result_lift[1:8]

    # ------------------ 状态机循环 ------------------
    
    # 定义状态
    STATE_APPROACH = 0  # 接近
    STATE_GRASP = 1     # 闭合夹爪
    STATE_LIFT = 2      # 抬起
    STATE_DONE = 3      # 完成
    
    current_state = STATE_APPROACH
    grasp_timer = 0     # 抓取等待计时器
    
    # 初始化第一个轨迹 (从当前位置 -> 抓取位置)
    current_joints = data.qpos[:7].copy()
    traj = JointSpaceTrajectory(start_joints=current_joints, end_joints=q_grasp, steps=800) # 800步走完

    print("\n=== 仿真开始 ===")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer_init(viewer)
        
        while viewer.is_running():
            # 1. 轨迹步进逻辑
            if not traj.finished:
                target_j = traj.get_next_waypoint()
                data.ctrl[:7] = target_j
            
            # 2. 状态机逻辑
            
            # --- 状态 0: 接近与检测 ---
            if current_state == STATE_APPROACH:
                # 保持夹爪张开
                data.ctrl[7] = 250
                
                # 检测距离
                site_pos = data.site("attachment_site").xpos
                cube_pos = data.body("cube").xpos
                dist = np.linalg.norm(site_pos - cube_pos)
                
                # 如果距离极小，且轨迹已经走得差不多了，进入抓取
                if dist < 0.02 and traj.current_step > (traj.steps * 0.9):
                    print(f"--> 接触物体 (距离 {dist:.3f}m)，开始抓取...")
                    current_state = STATE_GRASP
                    grasp_timer = 0 # 重置计时器

            # --- 状态 1: 闭合夹爪 ---
            elif current_state == STATE_GRASP:
                # 闭合夹爪 (设为 0)
                data.ctrl[7] = 0.0
                
                # 倒计时等待 (给物理引擎一点时间去计算接触力)
                grasp_timer += 1
                if grasp_timer > 100: # 等待 100 个仿真步 (约 0.2秒)
                    print("--> 抓取稳固，开始抬升！")
                    current_state = STATE_LIFT
                    
                    # 【关键】重新规划轨迹：从当前姿态 -> 抬起姿态
                    curr_j = data.qpos[:7].copy()
                    traj = JointSpaceTrajectory(start_joints=curr_j, end_joints=q_lift, steps=800)

            # --- 状态 2: 抬起 ---
            elif current_state == STATE_LIFT:
                # 保持夹爪闭合
                data.ctrl[7] = 0.0
                
                if traj.finished:
                    print("--> 抬升完成！")
                    current_state = STATE_DONE

            # --- 状态 3: 保持 ---
            elif current_state == STATE_DONE:
                pass # 保持最后的状态不动

            # 3. 物理引擎步进
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.002)

if __name__ == "__main__":
    main()