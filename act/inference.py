import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18
import mujoco.viewer
import mujoco
import numpy as np
import time
import pickle
import os

# ================= 配置 =================
CKPT_DIR = "model"
MODEL_PATH = os.path.join(CKPT_DIR, "policy_last.pth") # 加载最后一次训练的模型
STATS_PATH = os.path.join(CKPT_DIR, "dataset_stats.pkl") # 加载统计量
XML_PATH = "franka_emika_panda/scene.xml"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHUNK_SIZE = 100  
SIM_DT = 0.002    
CONTROL_DT = 0.02 

# ================= 1. 模型定义 (与 train_v2.py 保持完全一致) =================
class ACTModel(nn.Module):
    def __init__(self, state_dim=9, action_dim=8, hidden_dim=256):
        super().__init__()
        # weights=None 避免推理时联网下载权重
        resnet = resnet18(weights=None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.img_proj = nn.Linear(512, hidden_dim)

        self.state_proj = nn.Linear(state_dim, hidden_dim)

        # CVAE Encoder 部分 (推理时不用，但为了加载权重必须定义)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.action_encoder = nn.Linear(action_dim, hidden_dim) 
        self.latent_proj = nn.Linear(hidden_dim, 2 * hidden_dim) 

        # Transformer Decoder
        self.transformer = nn.Transformer(
            d_model=hidden_dim, nhead=4, num_encoder_layers=2, 
            num_decoder_layers=2, dim_feedforward=512, batch_first=True
        )
        
        self.pos_embed = nn.Parameter(torch.zeros(CHUNK_SIZE, hidden_dim))
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.latent_out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, img, qpos, actions=None, is_pad=None):
        B = img.shape[0]

        # 1. 特征提取
        # [关键修复] 使用 flatten(1) 而不是 squeeze()，防止 Batch=1 时报错
        img_embed = self.backbone(img).flatten(1) # (B, 512)
        img_embed = self.img_proj(img_embed).unsqueeze(1) # (B, 1, 256)
        state_embed = self.state_proj(qpos).unsqueeze(1)  # (B, 1, 256)

        # 2. CVAE 推理逻辑
        # 推理时没有 actions，Z 直接设为 0 (均值)
        z = torch.zeros(B, 256).to(img.device)

        # 3. Transformer Decoder
        z_embed = self.latent_out_proj(z).unsqueeze(1)
        src = torch.cat([z_embed, img_embed, state_embed], dim=1) # Input Sequence
        
        query_embed = self.pos_embed.unsqueeze(0).repeat(B, 1, 1) # Output Query
        
        output = self.transformer(src, query_embed)
        pred_actions = self.action_head(output)
        
        return pred_actions, None, None

# ================= 2. 辅助函数 =================
def load_stats(path):
    with open(path, 'rb') as f:
        stats = pickle.load(f)
    # 把 numpy 转成 tensor 并放到 GPU 上
    for k, v in stats.items():
        stats[k] = v.to(DEVICE)
    return stats

def get_pixel_obs(renderer, data):
    """获取图像并预处理"""
    imgs = []
    for cam in ['top_camera', 'wrist_camera']: 
        renderer.update_scene(data, camera=cam)
        imgs.append(renderer.render())
    
    top_img = imgs[0] # 只用 top
    
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
    # Home Pose
    home_pose = [0, -0.785, 0, -2.356, 0, 1.571, 0.785] 
    data.qpos[:7] = home_pose
    data.qpos[7] = 0.04; data.qpos[8] = 0.04 
    data.ctrl[:7] = home_pose; data.ctrl[7] = 255 

    cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube")
    if cube_id != -1:
        adr = model.jnt_qposadr[cube_id]
        # ⚠️ 这里保持和训练时一样的随机分布，或者稍小一点方便测试
        rx = np.random.uniform(0.45, 0.65)
        ry = np.random.uniform(-0.2, 0.2)
        data.qpos[adr] = rx; data.qpos[adr+1] = ry; data.qpos[adr+2] = 0.03
        data.qpos[adr+3:adr+7] = [1, 0, 0, 0]

    for _ in range(100): mujoco.mj_step(model, data)

# ================= 3. 主程序 =================
def main():
    # 1. 加载统计量 (翻译官)
    print(f"正在加载统计量: {STATS_PATH}")
    stats = load_stats(STATS_PATH)
    
    # 2. 加载模型
    print(f"正在加载模型: {MODEL_PATH}")
    model = ACTModel().to(DEVICE)
    # 加载权重 (weights_only=True 更安全)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()
    print("模型就绪！")

    # 3. 环境初始化
    model_mj = mujoco.MjModel.from_xml_path(XML_PATH)
    data_mj = mujoco.MjData(model_mj)
    renderer = mujoco.Renderer(model_mj, height=480, width=640)
    n_substeps = int(CONTROL_DT / SIM_DT)

    with mujoco.viewer.launch_passive(model_mj, data_mj) as viewer:
        viewer.cam.lookat[:] = [0.5, 0, 0.2]
        viewer.cam.distance = 1.5; viewer.cam.azimuth = 130; viewer.cam.elevation = -30
        
        while viewer.is_running():
            print("\n>>> 重置环境...")
            reset_env_random(model_mj, data_mj)
            viewer.sync()
            time.sleep(1)

            step = 0
            max_steps = 400 
            
            # 滚动时域控制: 预测100步，只走前K步，然后重新看、重新预测
            # 这样如果抓歪了，下一次预测能修回来
            EXECUTION_HORIZON = 20 

            while step < max_steps:
                # --- A. 获取观测 ---
                img_tensor = get_pixel_obs(renderer, data_mj)
                
                # 获取 qpos (9维)
                qpos_raw = torch.from_numpy(data_mj.qpos[:9].copy()).float().to(DEVICE)
                
                # 【关键步骤 1】: 归一化 Input (Translate to Model Language)
                # qpos = (raw - mean) / std
                qpos_norm = (qpos_raw - stats['qpos_mean']) / stats['qpos_std']
                qpos_norm = qpos_norm.unsqueeze(0) # 增加 Batch 维度 -> (1, 9)

                # --- B. 模型预测 ---
                with torch.no_grad():
                    # 推理时 actions=None, is_pad=None
                    all_actions_norm, _, _ = model(img_tensor, qpos_norm, None, None)
                    all_actions_norm = all_actions_norm[0] # 取出 batch 0 -> (100, 8)

                # 【关键步骤 2】: 反归一化 Output (Translate back to Reality)
                # action = norm * std + mean
                all_actions = all_actions_norm * stats['action_std'] + stats['action_mean']
                all_actions = all_actions.cpu().numpy()

                # --- C. 执行动作 (滚动执行) ---
                # print(f"Step {step}: 预测完成，执行前 {EXECUTION_HORIZON} 步...")
                
                for i in range(EXECUTION_HORIZON):
                    target_ctrl = all_actions[i]
                    
                    # 发送指令
                    data_mj.ctrl[:8] = target_ctrl
                    
                    # 物理步进
                    for _ in range(n_substeps):
                        mujoco.mj_step(model_mj, data_mj)
                    
                    viewer.sync()
                    step += 1
                    if step >= max_steps: break
                
                if step >= max_steps: break

            print("本集结束")
            time.sleep(0.5)

if __name__ == "__main__":
    main()