# Mujoco IL — Panda 机械臂模仿学习实践

## 简介
这是一个基于 MuJoCo 的 Franka Emika Panda 机械臂模仿学习（Imitation Learning, IL）实践项目。当前实现了基于 ACT（Action Transformer / CVAE 风格）算法的训练/推理 pipeline，并包含一个抓取任务作为示范。项目为后续添加更多算法（例如 DDPG/DP）、更多任务（如物体排序、移动到目标位置等）提供基础框架。


![ACT 抓取示例](docs/act_pick.gif)

## 目标
- 提供一个清晰的模仿学习流水线（数据记录、训练、重训练、推理/验证）。
- 以抓取任务为起点，逐步扩展更多任务与算法。
- 保持代码可复现、可复用，并支持大模型文件的合理存储（例如 Git LFS 或 Releases）。

## 项目结构（摘要）
- `act/`：基于 ACT 的实现代码
  - `run/`：训练与运行脚本
    - `1.record_ik.py`、`1.record_mocap.py`：数据记录脚本
    - `2.check_data.py`、`2.visualization_data.py`：数据检查与可视化
    - `3.train.py`、`3.retrain.py`：训练与重训练脚本
    - `4.inference.py`、`4.inference_video.py`、`4.validate_model.py`：推理/验证脚本
- `franka_emika_panda/`：模型与场景描述（XML/URDF）和资源文件

## 依赖
建议使用 conda 创建隔离环境（示例）：

```powershell
conda create -n rl_robot python=3.10 -y
conda activate rl_robot
pip install -r requirements.txt
# 额外：渲染/视频相关
pip install imageio[ffmpeg]
```

（请根据 `requirements.txt` 安装所需包；MuJoCo 的 Python 绑定与渲染在不同平台上可能需要额外配置。）

## 快速开始
1. 准备环境与依赖（见上）。
2. 运行数据记录（示例）:

```powershell
cd F:\mujoco_il
python act/run/1.record_mocap.py
```

3. 训练/重训练（示例）:

```powershell
python act/run/3.train.py
# 或重训练
python act/run/3.retrain.py
```

4. 推理 / 可视化

```powershell
# 交互式选择模型或在代码中指定 MODEL_CHOICE
python act/run/4.inference.py
# 录制视频/更高级的验证
python act/run/4.inference_video.py
python act/run/4.validate_model.py
```

## 模型选择（在 `4.inference.py`）
- 有两种方式选择要加载的 checkpoint：
  - 在代码中直接指定：编辑 `act/run/4.inference.py` 中的 `MODEL_CHOICE = "policy_epoch_50.pth"`（相对于 `model/` 目录）。
  - 交互式选择：运行 `4.inference.py` 时会列出 `model/` 下所有 `.pth` 文件，输入序号或文件名选择，回车会选择最新的文件。

## 未来计划
- 添加更多算法（例如基于强化学习的 DP、DDPG 等）与训练策略。
- 增加更多任务：物体排序、放置、复杂抓取场景等。
- 增加单元测试与 CI，自动化模型发布（Release + 下载脚本）。



## 贡献与联系
- 欢迎提交 issue/PR，或把你的任务/算法添加为新的子目录。
- 如果你想加入项目，请通过邮件联系我（zongyangwu2024@163.com）。



