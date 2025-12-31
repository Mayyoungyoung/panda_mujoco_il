import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import h5py
import os
import pickle
import torchvision.transforms as transforms
from torchvision.models import resnet18
from tqdm import tqdm  # 进度条库

# ================= 配置 =================
DATA_DIR = "../../data_act"        # 数据目录
CKPT_DIR = "model_finetune"        # 【建议修改】建议改为新的文件夹名字（例如 model_v2），这样绝对安全，不会覆盖旧模型
                                   # 如果你坚持用原来的 "model" 文件夹，请务必设置正确的 START_EPOCH

# --- 核心修改：预训练/断点续训设置 ---
RESUME_CHECKPOINT = "model/policy_last.pth"  # 【新增】这里填你之前训练好的模型路径。如果填 None，则从头训练
START_EPOCH = 20                             # 【新增】建议填之前的轮数（例如 50），这样保存的文件名会顺延

BATCH_SIZE = 128               # 4090 显卡建议 128
NUM_EPOCHS = 80                # 接着再训练多少轮
LR = 1e-5                      # 学习率 (微调建议 1e-5，防止破坏已有能力)
CHUNK_SIZE = 60                # 预测未来 60 步 (缩短步长以增加稳定性)
KL_WEIGHT = 20                 # CVAE KL 散度权重 (增加权重以减少抖动)
NUM_WORKERS = 16               # 4090 显卡建议 8 或 16

# ================= 0. 设备检测与打印 =================
def get_device():
    if torch.cuda.is_available():
        d = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        print(f"\n✅ 检测到 GPU: {name}")
        print(f"🚀 将使用 CUDA 进行加速训练\n")
    else:
        d = torch.device("cpu")
        print(f"\n⚠️ 未检测到 GPU，正在使用 CPU")
        print(f"🐢 训练速度可能会较慢\n")
    return d

DEVICE = get_device()

if not os.path.exists(CKPT_DIR):
    os.makedirs(CKPT_DIR)

# ================= 1. 数据预处理工具 =================
def get_dataset_stats(data_dir):
    """统计整个数据集的 Mean 和 Std"""
    # 优先检查 CKPT_DIR 下是否有统计量
    stats_path = os.path.join(CKPT_DIR, "dataset_stats.pkl")
    
    # 尝试去 RESUME_CHECKPOINT 所在的文件夹找统计量（防止重复计算）
    if not os.path.exists(stats_path) and RESUME_CHECKPOINT:
        old_dir = os.path.dirname(RESUME_CHECKPOINT)
        old_stats = os.path.join(old_dir, "dataset_stats.pkl")
        if os.path.exists(old_stats):
            print(f"🔄 从旧模型目录复制统计量: {old_stats}")
            with open(old_stats, 'rb') as f:
                stats = pickle.load(f)
            # 保存到新目录
            with open(stats_path, 'wb') as f:
                pickle.dump(stats, f)
            return stats

    if os.path.exists(stats_path):
        print(f"🔄 发现已存在的统计量文件: {stats_path}，直接加载...")
        with open(stats_path, 'rb') as f:
            return pickle.load(f)

    print("📊 未发现统计量，正在重新计算 (Normalization Stats)...")
    all_qpos = []
    all_action = []
    
    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.hdf5')]
    
    for f_path in tqdm(files):
        with h5py.File(f_path, 'r') as f:
            all_qpos.append(f['/observations/qpos'][()])
            all_action.append(f['/action'][()])
    
    # 拼接所有数据
    all_qpos = np.concatenate(all_qpos, axis=0)
    all_action = np.concatenate(all_action, axis=0)
    
    # 计算均值和标准差
    stats = {
        'qpos_mean': torch.from_numpy(np.mean(all_qpos, axis=0)).float(),
        'qpos_std': torch.from_numpy(np.std(all_qpos, axis=0)).float(),
        'action_mean': torch.from_numpy(np.mean(all_action, axis=0)).float(),
        'action_std': torch.from_numpy(np.std(all_action, axis=0)).float()
    }
    
    # 防止 std 为 0
    stats['qpos_std'] = torch.clip(stats['qpos_std'], 1e-2, None)
    stats['action_std'] = torch.clip(stats['action_std'], 1e-2, None)
    
    # 保存
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)
    
    print("✅ 统计完成并保存！")
    return stats

# ================= 2. 数据集类 =================
class ACTDataset(Dataset):
    def __init__(self, data_dir, stats):
        self.stats = stats
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # --- 构建索引 ---
        self.indices = []
        files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.hdf5')])
        
        print("📂 正在扫描数据集，构建索引...")
        for f_path in tqdm(files):
            with h5py.File(f_path, 'r') as f:
                # 获取该 Episode 的总帧数
                total_frames = f['/action'].shape[0]
                # 将每一帧都加入索引
                for i in range(total_frames):
                    self.indices.append((f_path, i))
        
        print(f"🎉 数据集构建完成! 共有 {len(files)} 个文件，展开为 {len(self.indices)} 个训练样本。")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        file_path, start_ts = self.indices[idx]
        
        with h5py.File(file_path, 'r') as f:
            # 1. 读取当前帧观测
            qpos = torch.from_numpy(f['/observations/qpos'][start_ts]).float()
            img_top = f['/observations/images/top'][start_ts] 
            
            # 2. 读取未来动作序列
            action_len = f['/action'].shape[0]
            end_ts = start_ts + CHUNK_SIZE
            
            if end_ts <= action_len:
                action = torch.from_numpy(f['/action'][start_ts:end_ts]).float()
                is_pad = torch.zeros(CHUNK_SIZE)
            else:
                real_len = action_len - start_ts
                action_real = torch.from_numpy(f['/action'][start_ts:]).float()
                last_action = action_real[-1].unsqueeze(0)
                pad_len = CHUNK_SIZE - real_len
                action_pad = last_action.repeat(pad_len, 1)
                
                action = torch.cat([action_real, action_pad], dim=0)
                is_pad = torch.cat([torch.zeros(real_len), torch.ones(pad_len)], dim=0)

        # 3. 归一化
        qpos = (qpos - self.stats['qpos_mean']) / self.stats['qpos_std']
        action = (action - self.stats['action_mean']) / self.stats['action_std']

        # 4. 图像变换
        img_tensor = self.transform(img_top)

        return img_tensor, qpos, action, is_pad

# ================= 3. ACT 模型 =================
class ACTModel(nn.Module):
    def __init__(self, state_dim=9, action_dim=8, hidden_dim=256):
        super().__init__()
        # 1. 视觉 Encoder
        resnet = resnet18(weights='DEFAULT')
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.img_proj = nn.Linear(512, hidden_dim)

        # 2. 状态 Encoder
        self.state_proj = nn.Linear(state_dim, hidden_dim)

        # 3. CVAE Encoder
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.action_encoder = nn.Linear(action_dim, hidden_dim) 
        self.latent_proj = nn.Linear(hidden_dim, 2 * hidden_dim) 

        # 4. 策略 Decoder
        self.transformer = nn.Transformer(
            d_model=hidden_dim, nhead=4, num_encoder_layers=2, 
            num_decoder_layers=2, dim_feedforward=512, batch_first=True
        )
        
        self.pos_embed = nn.Parameter(torch.randn(CHUNK_SIZE, hidden_dim))
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.latent_out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, img, qpos, actions=None, is_pad=None):
        B = img.shape[0]

        # 特征提取
        img_embed = self.backbone(img).flatten(1) # (B, 512)
        img_embed = self.img_proj(img_embed).unsqueeze(1) # (B, 1, 256)
        state_embed = self.state_proj(qpos).unsqueeze(1)  # (B, 1, 256)

        # CVAE
        mu, logvar = None, None
        if actions is not None:
            # 训练模式
            action_embed = self.action_encoder(actions) 
            action_summary = torch.mean(action_embed, dim=1, keepdim=True)
            encoder_input = torch.cat([self.cls_token.repeat(B, 1, 1), state_embed, action_summary], dim=1)
            combined_feat = torch.mean(encoder_input, dim=1) 
            
            latent_dist = self.latent_proj(combined_feat)
            mu = latent_dist[:, :256]
            logvar = latent_dist[:, 256:]
            
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            # 推理模式
            z = torch.zeros(B, 256).to(img.device)

        # Decoder
        z_embed = self.latent_out_proj(z).unsqueeze(1)
        src = torch.cat([z_embed, img_embed, state_embed], dim=1)
        query_embed = self.pos_embed.unsqueeze(0).repeat(B, 1, 1)
        
        # 修正：明确传入 src 和 tgt
        output = self.transformer(src=src, tgt=query_embed)
        pred_actions = self.action_head(output)
        
        return pred_actions, mu, logvar

# ================= 4. 损失函数 =================
def compute_loss(pred_actions, true_actions, is_pad, mu, logvar):
    # L1 Loss
    all_l1 = nn.functional.l1_loss(pred_actions, true_actions, reduction='none')
    l1 = (all_l1 * (1 - is_pad.unsqueeze(-1))).mean()

    # KL Divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss = kl_loss / pred_actions.shape[0]

    total_loss = l1 + KL_WEIGHT * kl_loss
    return total_loss, l1, kl_loss

# ================= 5. 训练循环 =================
def train():
    # 1. 准备统计量
    stats = get_dataset_stats(DATA_DIR)

    # 2. 准备数据加载器
    dataset = ACTDataset(DATA_DIR, stats)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True) 
    
    # 3. 初始化模型
    model = ACTModel().to(DEVICE)
    
    # --- 核心修改：加载之前的权重并处理尺寸不匹配 ---
    if RESUME_CHECKPOINT is not None and os.path.exists(RESUME_CHECKPOINT):
        print(f"\n🔄 正在加载预训练权重: {RESUME_CHECKPOINT}")
        try:
            # 加载权重字典
            state_dict = torch.load(RESUME_CHECKPOINT, map_location=DEVICE)
            
            # -----------------------------------------------------------
            # 【关键修改】智能权重裁剪逻辑
            # -----------------------------------------------------------
            if 'pos_embed' in state_dict:
                loaded_len = state_dict['pos_embed'].shape[0]
                current_len = model.pos_embed.shape[0]
                
                if loaded_len != current_len:
                    print(f"⚠️ 检测到预测步长 (Chunk Size) 变化: 旧模型 {loaded_len} -> 新模型 {current_len}")
                    
                    if loaded_len > current_len:
                        print(f"✂️ 正在自动裁剪 pos_embed 权重，保留前 {current_len} 步...")
                        # 核心动作：切片取前 N 个
                        state_dict['pos_embed'] = state_dict['pos_embed'][:current_len]
                    else:
                        print(f"❌ 错误: 新的 CHUNK_SIZE ({current_len}) 比旧模型 ({loaded_len}) 大，无法进行权重裁剪！")
                        print("建议：将 CHUNK_SIZE 改回原来的大小，或者从头训练。")
                        return

            # 加载处理后的权重
            model.load_state_dict(state_dict) # 默认 strict=True
            print("✅ 权重加载成功（已适配新步长）！将在该模型基础上继续训练。")
            
        except Exception as e:
            print(f"❌ 加载权重失败: {e}")
            import traceback
            traceback.print_exc()
            return
    else:
        if RESUME_CHECKPOINT is not None:
             print(f"\n⚠️ 警告: 指定的路径 {RESUME_CHECKPOINT} 不存在，将从头开始训练。")
        else:
             print("\n🆕 未指定预训练模型，将从头开始训练。")

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    print(f"\n🚀 开始训练! 总样本数: {len(dataset)}, Batch Size: {BATCH_SIZE}")
    print(f"📅 起始 Epoch: {START_EPOCH + 1}, 计划训练 Epochs: {NUM_EPOCHS}")
    
    model.train()
    
    # --- Epoch 循环 ---
    for epoch in range(START_EPOCH, START_EPOCH + NUM_EPOCHS):
        total_l1 = 0
        total_kl = 0
        
        loop = tqdm(dataloader, leave=False)
        for img, qpos, action, is_pad in loop:
            img = img.to(DEVICE)
            qpos = qpos.to(DEVICE)
            action = action.to(DEVICE)
            is_pad = is_pad.to(DEVICE)
            
            optimizer.zero_grad()
            
            pred_actions, mu, logvar = model(img, qpos, action, is_pad)
            
            loss, l1, kl = compute_loss(pred_actions, action, is_pad, mu, logvar)
            
            loss.backward()
            optimizer.step()
            
            total_l1 += l1.item()
            total_kl += kl.item()
            
            loop.set_description(f"Epoch {epoch+1}/{START_EPOCH + NUM_EPOCHS}")
            loop.set_postfix(l1=l1.item(), kl=kl.item())
        
        scheduler.step()
        
        avg_l1 = total_l1 / len(dataloader)
        avg_kl = total_kl / len(dataloader)
        
        print(f"Epoch {epoch+1} | L1 Loss: {avg_l1:.5f} | KL Loss: {avg_kl:.5f}")
            
        # 每 10 轮保存一次权重
        if (epoch+1) % 10 == 0:  # 这里我改成每10轮保存一次，方便微调
            ckpt_path = os.path.join(CKPT_DIR, f"policy_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"💾 模型已保存: {ckpt_path}")

    # 保存最终模型
    torch.save(model.state_dict(), os.path.join(CKPT_DIR, "policy_last.pth"))
    print("\n✅ 训练全部完成！最终模型已保存。")

if __name__ == "__main__":
    train()