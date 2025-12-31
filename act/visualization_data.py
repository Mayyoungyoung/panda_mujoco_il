import h5py
import cv2
import numpy as np
import os
import time

# ================= 配置 =================
DATA_DIR = "data_act"   # 你的数据目录
PLAY_SPEED = 1.0        # 播放倍速 (0.5=慢放, 1.0=正常, 2.0=快进)

def visualize_episode(dataset_dir, episode_idx):
    """返回值: 'next'=下一个, 'prev'=上一个, 'quit'=退出"""
    file_path = os.path.join(dataset_dir, f"episode_{episode_idx}.hdf5")
    
    if not os.path.exists(file_path):
        print(f"❌ 找不到文件: {file_path}")
        return 'next'  # 跳过不存在的文件

    print(f"\n{'='*60}")
    print(f"📁 Episode {episode_idx}: {file_path}")
    print(f"{'='*60}")
    
    with h5py.File(file_path, 'r') as root:
        # 1. 获取数据长度
        # 检查是否是 ACT 官方格式 (observations/images/...)
        is_sim = root.attrs.get('sim', False)
        qpos = root['/observations/qpos'][()]
        qvel = root['/observations/qvel'][()]
        action = root['/action'][()]
        
        # 图像数据通常比较大，建议按帧读取，不要一次性全部读入内存(如果是超大数据集)
        # 但对于几十秒的 episode，一次读出来也没事
        image_dict = root['/observations/images']
        top_imgs = image_dict['top'][()]
        wrist_imgs = image_dict['wrist'][()]
        num_steps = len(qpos)
        print(f"📊 总帧数: {num_steps}")
        print(f"📐 Qpos Shape: {qpos.shape}")
        print(f"🎮 Action Shape: {action.shape}")
        
        print("\n⌨️  快捷键:")
        print("  Space - 暂停/继续")
        print("  N     - 跳到下一个 Episode")
        print("  P     - 跳到上一个 Episode")  
        print("  Q     - 退出程序")
        print("  A/D   - 上一帧/下一帧 (暂停时)")
        print(f"{'='*60}\n")

        idx = 0
        paused = False
        
        while idx < num_steps:
            # --- 1. 图像处理 ---
            # 这里的图像是 RGB，OpenCV 需要 BGR
            top_img = cv2.cvtColor(top_imgs[idx], cv2.COLOR_RGB2BGR)
            wrist_img = cv2.cvtColor(wrist_imgs[idx], cv2.COLOR_RGB2BGR)
            
            # 简单拼接 (横向)
            # 如果高度不一样需要 resize，这里假设都是 480x640
            canvas = np.hstack([top_img, wrist_img])
            
            # --- 2. 数据叠加 ---
            # 获取当前机械臂状态
            # 假设 qpos 前7位是关节，第8位是夹爪 (Panda: 0=Close, 0.04=Open)
            curr_qpos = qpos[idx]
            curr_action = action[idx]
            
            # 判断夹爪状态 (Panda 夹爪全开是 0.04 * 2 = 0.08，或者单指 0.04)
            # 根据你的数据，第7个索引(从0开始)通常是夹爪
            # 如果是两个指头，可能是 idx 7 和 8
            gripper_val = curr_qpos[7] 
            gripper_state = "OPEN" if gripper_val > 0.03 else "CLOSED"
            color = (0, 255, 0) if gripper_state == "OPEN" else (0, 0, 255)

            # 在画面上打印信息
            info_text = [
                f"Episode: {episode_idx} | Frame: {idx}/{num_steps}",
                f"Gripper: {gripper_state} ({gripper_val:.4f})",
                f"Action: {curr_action[7]:.4f}",
                f"[N] Next | [P] Prev | [Space] Pause | [Q] Quit"
            ]
            
            for i, line in enumerate(info_text):
                cv2.putText(canvas, line, (20, 40 + i*30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2) # 白字
                cv2.putText(canvas, line, (20, 40 + i*30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1)       # 彩色描边

            # --- 3. 显示与交互 ---
            cv2.imshow("ACT Data Inspector (Left: Top | Right: Wrist)", canvas)
            
            # 计算延迟: 50Hz = 20ms
            delay = int(20 / PLAY_SPEED)
            if delay < 1: delay = 1
            
            key = cv2.waitKey(0 if paused else delay)
            
            if key & 0xFF == ord('q'): # Quit
                cv2.destroyAllWindows()
                return 'quit'
            elif key & 0xFF == ord('n'): # Next episode
                cv2.destroyAllWindows()
                return 'next'
            elif key & 0xFF == ord('p'): # Previous episode
                cv2.destroyAllWindows()
                return 'prev'
            elif key & 0xFF == ord(' '): # Pause
                paused = not paused
            elif key & 0xFF == ord('d'): # Next frame (Debug)
                if paused and idx < num_steps - 1:
                    idx += 1
            elif key & 0xFF == ord('a'): # Prev frame (Debug)
                if paused and idx > 0:
                    idx -= 1
            
            if not paused:
                idx += 1

    cv2.destroyAllWindows()
    return 'next'  # 播放完自动跳到下一个

def get_episode_count(dataset_dir):
    """获取数据集中的 episode 数量"""
    count = 0
    while os.path.exists(os.path.join(dataset_dir, f"episode_{count}.hdf5")):
        count += 1
    return count

if __name__ == "__main__":
    total_episodes = get_episode_count(DATA_DIR)
    
    if total_episodes == 0:
        print(f"❌ 在 {DATA_DIR} 中没有找到任何 episode 文件")
        exit(1)
    
    print(f"🎬 找到 {total_episodes} 个 episode 文件")
    print(f"📂 数据目录: {DATA_DIR}\n")
    
    current_idx = 0
    
    while True:
        # 确保索引在有效范围内
        if current_idx < 0:
            current_idx = 0
        elif current_idx >= total_episodes:
            print(f"\n✅ 已经看完所有 {total_episodes} 个 episode！")
            break
        
        # 播放当前 episode
        result = visualize_episode(DATA_DIR, current_idx)
        
        if result == 'quit':
            print("\n👋 退出查看")
            break
        elif result == 'next':
            current_idx += 1
        elif result == 'prev':
            current_idx -= 1
    
    cv2.destroyAllWindows()
    print("程序结束")