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
DATA_DIR = "data_act"        # 数据目录
CKPT_DIR = "model"           # 模型保存目录
BATCH_SIZE = 16              # 建议设为 16 或 32，取决于显存大小
NUM_EPOCHS = 50              # 训练轮数 (因为数据量变大了，轮数可以适当减少，或者保持 100)
LR = 1e-4                    # 学习率
CHUNK_SIZE = 100             # 预测未来 100 步
KL_WEIGHT = 10               # CVAE KL 散度权重

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
    stats_path = os.path.join(CKPT_DIR, "dataset_stats.pkl")
    if os.path.exists(stats_path):
        print(f"🔄 发现已存在的统计量文件: {stats_path}，直接加载...")
        with open(stats_path, 'rb') as f:
            return pickle.load(f)

    print("📊 正在计算数据集统计量 (Normalization Stats)...")
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

# ================= 2. 数据集类 (核心修复版) =================
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
                # 数据足够长，直接切片
                action = torch.from_numpy(f['/action'][start_ts:end_ts]).float()
                is_pad = torch.zeros(CHUNK_SIZE)
            else:
                # 数据不够长，进行 Padding
                real_len = action_len - start_ts
                action_real = torch.from_numpy(f['/action'][start_ts:]).float()
                # 重复最后一步
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
        # 1. 视觉 Encoder (使用预训练权重可以加速收敛)
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
        
        output = self.transformer(src, query_embed)
        pred_actions = self.action_head(output)
        
        return pred_actions, mu, logvar

# ================= 4. 损失函数 =================
def compute_loss(pred_actions, true_actions, is_pad, mu, logvar):
    # L1 Loss (只计算非 Padding 部分)
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
    # Windows下 num_workers 设置为 0 比较稳妥，Linux 可以设置 4 或 8
    dataset = ACTDataset(DATA_DIR, stats)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0) 
    
    # 3. 初始化模型
    model = ACTModel().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    print(f"\n🚀 开始训练! 总样本数: {len(dataset)}, Batch Size: {BATCH_SIZE}")
    model.train()
    
    for epoch in range(NUM_EPOCHS):
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
            
            loop.set_description(f"Epoch {epoch+1}/{NUM_EPOCHS}")
            loop.set_postfix(l1=l1.item(), kl=kl.item())
        
        scheduler.step()
        
        avg_l1 = total_l1 / len(dataloader)
        avg_kl = total_kl / len(dataloader)
        
        # 每轮都打印一次 Log
        print(f"Epoch {epoch+1} | L1 Loss: {avg_l1:.5f} | KL Loss: {avg_kl:.5f}")
            
        # 每 10 轮保存一次权重
        if (epoch+1) % 10 == 0:
            ckpt_path = os.path.join(CKPT_DIR, f"policy_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"💾 模型已保存: {ckpt_path}")

    # 保存最终模型
    torch.save(model.state_dict(), os.path.join(CKPT_DIR, "policy_last.pth"))
    print("\n✅ 训练全部完成！最终模型已保存。")

if __name__ == "__main__":
    train()