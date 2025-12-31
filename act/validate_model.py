"""
训练后验证脚本：检查模型预测是否合理
"""
import torch
import numpy as np
import h5py
import pickle
import os
import sys
sys.path.append('.')
from train import ACTModel, ACTDataset

# ================= 配置 =================
DATA_DIR = "data_act"
CKPT_DIR = "model"
MODEL_PATH = os.path.join(CKPT_DIR, "policy_last.pth")
STATS_PATH = os.path.join(CKPT_DIR, "dataset_stats.pkl")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def validate():
    print("=" * 60)
    print("模型验证：检查预测是否合理")
    print("=" * 60)
    
    # 1. 加载统计量和数据
    with open(STATS_PATH, 'rb') as f:
        stats = pickle.load(f)
    
    dataset = ACTDataset(DATA_DIR, stats)
    
    # 2. 加载模型
    model = ACTModel().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()
    print(f"✅ 模型已加载: {MODEL_PATH}\n")
    
    # 3. 随机抽取3个样本验证
    indices = np.random.choice(len(dataset), min(3, len(dataset)), replace=False)
    
    for idx in indices:
        print(f"\n{'='*60}")
        print(f"样本 {idx}")
        print(f"{'='*60}")
        
        img, qpos, true_action, is_pad = dataset[idx]
        
        # 添加 batch 维度
        img = img.unsqueeze(0).to(DEVICE)
        qpos = qpos.unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            pred_action, mu, logvar = model(img, qpos, None, None)
        
        pred_action = pred_action[0].cpu().numpy()  # (100, 8)
        true_action = true_action.numpy()  # (100, 8)
        
        # 反归一化到真实空间
        pred_action_real = pred_action * stats['action_std'].numpy() + stats['action_mean'].numpy()
        true_action_real = true_action * stats['action_std'].numpy() + stats['action_mean'].numpy()
        
        # 计算误差（只看前10步）
        error = np.abs(pred_action_real[:10] - true_action_real[:10])
        
        print(f"\n前10步关节动作预测 vs 真实值:")
        print(f"{'Step':<6} {'Joint0_Pred':<12} {'Joint0_True':<12} {'Error':<10}")
        print("-" * 50)
        for i in range(10):
            print(f"{i:<6} {pred_action_real[i, 0]:<12.4f} {true_action_real[i, 0]:<12.4f} {error[i, 0]:<10.4f}")
        
        print(f"\n统计信息:")
        print(f"  平均误差: {error.mean():.4f}")
        print(f"  最大误差: {error.max():.4f}")
        print(f"  预测范围: [{pred_action_real.min():.4f}, {pred_action_real.max():.4f}]")
        print(f"  真实范围: [{true_action_real.min():.4f}, {true_action_real.max():.4f}]")
        
        # 检查预测是否"卡死"
        pred_std = pred_action_real[:20].std(axis=0).mean()
        true_std = true_action_real[:20].std(axis=0).mean()
        
        print(f"\n动作变化程度:")
        print(f"  预测动作标准差: {pred_std:.4f}")
        print(f"  真实动作标准差: {true_std:.4f}")
        
        if pred_std < 0.01:
            print("  ⚠️  警告：预测几乎没有变化，模型可能没有学习！")
        elif pred_std < true_std * 0.1:
            print("  ⚠️  警告：预测变化太小，模型过于保守！")
        else:
            print("  ✅ 预测有合理的变化")

if __name__ == "__main__":
    validate()
