import mujoco
import numpy as np
import cv2

# 1. 加载模型
model = mujoco.MjModel.from_xml_path("./franka_emika_panda/scene.xml")
renderer = mujoco.Renderer(model, height=480, width=640)
data = mujoco.MjData(model)

# 2. 初始化模拟
mujoco.mj_forward(model, data)

print("按 'q' 退出测试窗口...")

while True:
    # 让物理世界向前推演一步 (虽然现在没动，但在循环里是必须的)
    mujoco.mj_step(model, data)

    # --- 获取 Top Camera 图像 ---
    renderer.update_scene(data, camera="top_camera")
    top_img = renderer.render() # 返回的是 RGB 格式

    # --- 获取 Wrist Camera 图像 ---
    renderer.update_scene(data, camera="wrist_camera")
    wrist_img = renderer.render()

    # --- 拼接图像用于显示 ---
    # 将两个图像横向拼接 (Horizontal Stack)
    combined_img = np.hstack((top_img, wrist_img))

    # OpenCV 默认是 BGR，MuJoCo 是 RGB，需要转换颜色空间以便正确显示
    combined_img_bgr = cv2.cvtColor(combined_img, cv2.COLOR_RGB2BGR)

    cv2.imshow("ACT Camera Check (Left: Top | Right: Wrist)", combined_img_bgr)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()