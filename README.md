# Mujoco IL — Panda 机械臂模仿学习实践

## 简介
这是一个基于 MuJoCo 的 Franka Emika Panda 机械臂模仿学习（Imitation Learning, IL）实践项目。当前实现了基于 ACT（Action Transformer / CVAE 风格）算法的训练/推理 pipeline，并包含一个抓取任务作为示范。项目为后续添加更多算法（例如 DDPG/DP）、更多任务（如物体排序、移动到目标位置等）提供基础框架。

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
- `model/`：训练生成的模型与统计量（`.pth`、`dataset_stats.pkl`）

> 注意：`model/` 下的大文件默认在 `.gitignore` 中被忽略（参见仓库根目录的 `.gitignore`）。

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

## 关于模型文件未被上传到 GitHub 的原因
仓库根目录的 `.gitignore` 默认忽略如下文件类型：`*.pth`, `*.pt`, `*.pkl` 等，因此 `model/` 下的模型文件和统计量不会被 git 跟踪，也不会被推送到远程仓库。

建议的处理方式：
- 推荐：使用 Git LFS 来托管大模型文件：
  ```powershell
  git lfs install
  git lfs track "*.pth"
  git lfs track "*.pkl"
  git add .gitattributes
  git add model/policy_epoch_100.pth model/dataset_stats.pkl
  git commit -m "Add model files via Git LFS"
  git push -u origin <branch>
  ```
- 或者把模型上传到 GitHub Release / Google Drive / S3，并在仓库中保存下载脚本（推荐用于公开/分发）。
- 若确实要把模型直接加入仓库，可临时强制添加：
  ```powershell
  git add -f model/policy_epoch_100.pth model/dataset_stats.pkl
  git commit -m "Force add model files"
  git push
  ```
  但这会使仓库膨胀，不推荐长期使用。

## 未来计划
- 添加更多算法（例如基于强化学习的 DP、DDPG 等）与训练策略。
- 增加更多任务：物体排序、放置、复杂抓取场景等。
- 增加单元测试与 CI，自动化模型发布（Release + 下载脚本）。



我将把截图放到 `docs/` 并在 README 中引用它们。

## 演示视频

HTML5 嵌入（GitHub 在不同界面可能不直接内嵌播放，但会提供下载/预览）：

<video src="docs/ep3.mp4" controls width="720">你的浏览器不支持 video 标签，请点击此处下载： <a href="docs/ep3.mp4">下载视频</a></video>

或者直接提供链接：

[演示视频：ep3.mp4](docs/ep3.mp4)

上传说明：

1. 在本地把视频放到 `docs/ep3.mp4`（若不存在 `docs/` 目录请先创建）。
2. 提交并推送：

```powershell
git add docs/ep3.mp4 README.md
git commit -m "Add demo video ep3.mp4 and README reference"
git push -u origin $(git branch --show-current)
```

注意：如果视频较大，建议使用外部托管（GitHub Release、Google Drive 或 S3），或使用 Git LFS；如需我代为上传并配置 Git LFS，请告知。

## 贡献与联系
- 欢迎提交 issue/PR，或把你的任务/算法添加为新的子目录。
- 如果你想加入项目，请通过邮件联系我（zongyangwu2024@163.com）。