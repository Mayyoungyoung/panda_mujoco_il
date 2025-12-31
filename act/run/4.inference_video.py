import os
# 【核心修改】这两行必须在 import mujoco 之前！
# 强制使用 EGL 后端（无头渲染），不依赖 X11 窗口
os.environ['MUJOCO_GL'] = 'egl' 

# 也可以加上这一行，强制让 PyTorch 和 MuJoCo 都能看到显卡
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18
import mujoco  # <--- mujoco 必须在环境变量设置之后导入
import numpy as np
import time
import pickle
import imageio

# ... 后面的代码保持不变 ...

# ================= 配置 =================
CKPT_DIR = "model"
MODEL_PATH = os.path.join(CKPT_DIR, "policy_last.pth")
STATS_PATH = os.path.join(CKPT_DIR, "dataset_stats.pkl")
XML_PATH = "../franka_emika_panda/scene.xml"
SAVE_VIDEO_PATH = "inference_result.mp4"  # <--- 视频保存路径
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ⚠️ 关键设置：告诉 MuJoCo 使用 EGL 后端进行无头渲染 (Headless Rendering)
os.environ['MUJOCO_GL'] = 'egl'

CHUNK_SIZE = 100
SIM_DT = 0.002
CONTROL_DT = 0.02

# ================= 1. 模型定义 (保持不变) =================
class ACTModel(nn.Module):
    def __init__(self, state_dim=9, action_dim=8, hidden_dim=256):
        super().__init__()
        resnet = resnet18(weights=None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.img_proj = nn.Linear(512, hidden_dim)
        self.state_proj = nn.Linear(state_dim, hidden_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.action_encoder = nn.Linear(action_dim, hidden_dim)
        self.latent_proj = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.transformer = nn.Transformer(
            d_model=hidden_dim, nhead=4, num_encoder_layers=2,
            num_decoder_layers=2, dim_feedforward=512, batch_first=True
        )
        self.pos_embed = nn.Parameter(torch.zeros(CHUNK_SIZE, hidden_dim))
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.latent_out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, img, qpos, actions=None, is_pad=None):
        B = img.shape[0]
        img_embed = self.backbone(img).flatten(1)
        img_embed = self.img_proj(img_embed).unsqueeze(1)
        state_embed = self.state_proj(qpos).unsqueeze(1)
        z = torch.randn(B, 256).to(img.device) * 0.1
        z_embed = self.latent_out_proj(z).unsqueeze(1)
        src = torch.cat([z_embed, img_embed, state_embed], dim=1)
        query_embed = self.pos_embed.unsqueeze(0).repeat(B, 1, 1)
        output = self.transformer(src, query_embed)
        pred_actions = self.action_head(output)
        return pred_actions, None, None

# ================= 2. 辅助函数 =================
def load_stats(path):
    with open(path, 'rb') as f:
        stats = pickle.load(f)
    for k, v in stats.items():
        stats[k] = v.to(DEVICE)
    return stats

def get_pixel_obs(renderer, data):
    renderer.update_scene(data, camera='top_camera')
    top_img = renderer.render()
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(top_img).unsqueeze(0).to(DEVICE)
    return img_tensor

def reset_env_random(model, data):
    mujoco.mj_resetData(model, data)
    home_pose = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
    data.qpos[:7] = home_pose
    data.qpos[7] = 0.04; data.qpos[8] = 0.04
    data.ctrl[:7] = home_pose; data.ctrl[7] = 255

    cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube")
    if cube_id != -1:
        adr = model.jnt_qposadr[cube_id]
        rx = np.random.uniform(0.45, 0.65)
        ry = np.random.uniform(-0.2, 0.2)
        data.qpos[adr] = rx; data.qpos[adr+1] = ry; data.qpos[adr+2] = 0.03
        data.qpos[adr+3:adr+7] = [1, 0, 0, 0]

    for _ in range(100): mujoco.mj_step(model, data)

# ================= 3. 主程序 (多回合版) =================
def main():
    # 1. 准备
    print(f"正在加载统计量: {STATS_PATH}")
    stats = load_stats(STATS_PATH)
    
    print(f"正在加载模型: {MODEL_PATH}")
    model = ACTModel().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()
    print("模型就绪！")

    # 2. 环境初始化
    model_mj = mujoco.MjModel.from_xml_path(XML_PATH)
    data_mj = mujoco.MjData(model_mj)
    renderer = mujoco.Renderer(model_mj, height=480, width=640)
    n_substeps = int(CONTROL_DT / SIM_DT)

    # ================= 配置录制次数 =================
    NUM_EPISODES = 5  # 👈 这里修改你想录几次
    # ==============================================

    for episode_idx in range(NUM_EPISODES):
        print(f"\n🎬 正在录制第 {episode_idx + 1}/{NUM_EPISODES} 个视频...")
        
        # 每个回合开始前重置环境
        reset_env_random(model_mj, data_mj)
        
        frames = [] 
        step = 0
        max_steps = 400 
        EXECUTION_HORIZON = 20 

        while step < max_steps:
            # --- A. 获取观测 ---
            img_tensor = get_pixel_obs(renderer, data_mj)
            qpos_raw = torch.from_numpy(data_mj.qpos[:9].copy()).float().to(DEVICE)
            qpos_norm = (qpos_raw - stats['qpos_mean']) / stats['qpos_std']
            qpos_norm = qpos_norm.unsqueeze(0)

            # --- B. 模型预测 ---
            with torch.no_grad():
                all_actions_norm, _, _ = model(img_tensor, qpos_norm, None, None)
                all_actions_norm = all_actions_norm[0]

            # --- C. 反归一化 ---
            all_actions = all_actions_norm * stats['action_std'] + stats['action_mean']
            all_actions = all_actions.cpu().numpy()

            # --- D. 执行动作 ---
            for i in range(EXECUTION_HORIZON):
                target_ctrl = all_actions[i]
                data_mj.ctrl[:8] = target_ctrl
                
                for _ in range(n_substeps):
                    mujoco.mj_step(model_mj, data_mj)
                
                # 渲染当前帧
                renderer.update_scene(data_mj, camera='top_camera') 
                frame = renderer.render()
                frames.append(frame)

                step += 1
                if step >= max_steps: break
            
            if step >= max_steps: break

        # 保存当前回合的视频
        video_name = f"result_ep{episode_idx}.mp4"
        imageio.mimsave(video_name, frames, fps=30)
        print(f"✅ 第 {episode_idx + 1} 个视频已保存: {video_name}")

    print("\n🎉 所有录制完成！请下载查看。")

if __name__ == "__main__":
    main()
