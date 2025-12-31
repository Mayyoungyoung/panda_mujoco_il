import mujoco
import mujoco.viewer
import cv2
import numpy as np
import time

# 1. 加载模型
m = mujoco.MjModel.from_xml_path('./franka_emika_panda/scene.xml')
d = mujoco.MjData(m)

# 2. 创建渲染器 (Renderer)
# 这里的 480, 640 是你想要的图像分辨率
renderer = mujoco.Renderer(m, height=480, width=640)

# 3. 运行仿真循环
with mujoco.viewer.launch_passive(m, d) as viewer:
    while viewer.is_running():
        step_start = time.time()

        # 物理步进
        mujoco.mj_step(m, d)
        
        # --- 核心代码：获取相机画面 ---
        for camera_name in ("top_camera", "wrist_camera"):
            renderer.update_scene(d, camera=camera_name)
            rgb_image = renderer.render()
            bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            cv2.imshow(f"Camera View: {camera_name}", bgr_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # ---------------------------

        viewer.sync()
        
        # 控制帧率
        time_until_next_step = m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

cv2.destroyAllWindows()