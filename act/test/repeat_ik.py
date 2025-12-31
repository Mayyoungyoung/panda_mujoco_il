import mujoco.viewer
import mujoco
import time
import ikpy.chain
import transforms3d as tf
import numpy as np
import random

# ================= 辅助类与函数 (保持不变) =================

def viewer_init(viewer):
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    viewer.cam.lookat[:] = [0.5, 0, 0.2]
    viewer.cam.distance = 1.5
    viewer.cam.azimuth = 130
    viewer.cam.elevation = -30

class JointSpaceTrajectory:
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

def get_ik_solution(chain, target_pos, target_euler_deg, initial_guess):
    # 1. 角度转弧度
    target_euler_rad = np.radians(target_euler_deg)
    target_orientation = tf.euler.euler2mat(*target_euler_rad)
    
    # 2. 计算
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

# ================= 新增：环境重置函数 =================

def reset_env(model, data):
    """
    重置环境：
    1. 机械臂回 Home
    2. 方块随机分布
    3. 物理预热 (让方块落稳)
    """
    # 1. 重置数据 (清除速度、加速度等历史缓存)
    mujoco.mj_resetData(model, data)

    # 2. 机械臂归位 (Home Pose)
    home_pose = [0, -0.785, 0, -2.356, 0, 1.571, 0.785] 
    data.qpos[:7] = home_pose
    data.qpos[7] = 0.04 # 张开
    data.qpos[8] = 0.04 
    
    # 同步控制信号，防止一开始就猛烈甩动
    data.ctrl[:7] = home_pose
    data.ctrl[7] = 255 # 255代表张开控制信号

    # 3. 随机放置方块
    # 假设方块的 joint 名字叫 "cube"，如果不是请在 xml 查看
    cube_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube")
    if cube_jnt_id != -1:
        qpos_adr = model.jnt_qposadr[cube_jnt_id]
        
        # 随机范围 X[0.4, 0.6], Y[-0.15, 0.15]
        rand_x = random.uniform(0.4, 0.7)
        rand_y = random.uniform(-0.3, 0.3)
        
        # 设置位置 (Z=0.03 离地1cm左右，保证不穿模但快速落地)
        data.qpos[qpos_adr] = rand_x
        data.qpos[qpos_adr+1] = rand_y
        data.qpos[qpos_adr+2] = 0.03 
        
        # 重置旋转 (防止方块生成时是歪的)
        data.qpos[qpos_adr+3:qpos_adr+7] = [1, 0, 0, 0] # Quaternion w,x,y,z

    # 4. 【关键步骤】物理预热 (Settle)
    # 不渲染画面，单纯跑物理计算，让方块真正落到桌子上
    for _ in range(50): 
        mujoco.mj_step(model, data)

    return True

# ================= 主程序 =================
def check_attachment(data):
    # 检查夹爪是否抓取到方块
    attachment_site_pos = data.site("attachment_site").xpos
    cube_pos = data.body("cube").xpos
    dist = np.linalg.norm(attachment_site_pos - cube_pos)
    return dist < 0.01  # 假设小于2cm为抓取成功

def main():
    # 1. 加载模型
    # 请确保路径正确
    model = mujoco.MjModel.from_xml_path('franka_emika_panda/scene.xml')
    data = mujoco.MjData(model)

    # 2. 初始化 IK 链
    my_chain = ikpy.chain.Chain.from_urdf_file(
        "franka_emika_panda/panda.urdf",
        base_elements=["panda_link0"],
        last_link_vector=[0, 0, 0.107], # 保持你调好的偏移
        active_links_mask=[False, True, True, True, True, True, True, True, False, False]
    )

    # 定义状态枚举
    STATE_INIT = 0      # 初始化/重置
    STATE_APPROACH = 1  # 接近
    STATE_GRASP = 2     # 抓取
    STATE_LIFT = 3      # 抬起

    # 初始状态
    current_state = STATE_INIT
    state_timer = 0
    traj = None
    q_grasp = None
    q_lift = None

    print("\n=== 循环抓取仿真开始 ===")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer_init(viewer)
        
        while viewer.is_running():
            if current_state == STATE_INIT:
                # 重置环境
                reset_env(model, data)
                cube_pos = data.body("cube").xpos
                current_state = STATE_APPROACH
                grasp_pos = cube_pos
                # 姿态：垂直向下 (180度翻转)
                grasp_euler = [180, 0, 45]
                
                # 初始猜测 (关键：J4弯曲)
                ref_pos_guess = [0.0] * 10
                ref_pos_guess[4] = -1.57
                ref_pos_guess[6] = 1.57

                # 计算 A 的解
                ik_result_grasp = get_ik_solution(my_chain, grasp_pos, grasp_euler, ref_pos_guess)
                if ik_result_grasp is None: return
                
                # 提取 7个关节角
                q_grasp = ik_result_grasp[1:8]

                # 动作 B: 抬起点 (Z轴抬高)
                lift_pos = cube_pos + np.array([0, 0, 0.5])
                lift_euler = [180, 0, 0]
                
                # 【技巧】用抓取姿态作为抬起姿态的初始猜测，保证动作连续性
                ref_pos_lift = ik_result_grasp.copy()
                
                # 计算 B 的解
                ik_result_lift = get_ik_solution(my_chain, lift_pos, lift_euler, ref_pos_lift)
                if ik_result_lift is None: return
                
                q_lift = ik_result_lift[1:8]

                # 初始化第一个轨迹 (从当前位置 -> 抓取位置)
                current_joints = data.qpos[:7].copy()
                traj = JointSpaceTrajectory(start_joints=current_joints, end_joints=q_grasp, steps=800) # 800步走完
           
            if not traj.finished:
                target_j = traj.get_next_waypoint()
                data.ctrl[:7] = target_j
            
            # --- 状态 0: 接近与检测 ---
            if current_state == STATE_APPROACH:
                # 保持夹爪张开
                data.ctrl[7] = 250
                
                # 如果距离极小，且轨迹已经走得差不多了，进入抓取
                if check_attachment(data) and traj.current_step > (traj.steps * 0.9):
                    print(f"--> 接触物体，开始抓取...")
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
                    current_state = STATE_INIT
                    for wait_step in range(50):
                        mujoco.mj_step(model, data)
                        viewer.sync()
                        time.sleep(0.002)
                    

            # 3. 物理引擎步进
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.002)

if __name__ == "__main__":
    main()