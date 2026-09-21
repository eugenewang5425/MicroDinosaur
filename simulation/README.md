# MicroDinosaur 仿真与训练

2026-09-21 整理。这里保存训练源码快照、精选实验和最新动作复核。本次整理包含一次有边界的起身续训，但不代表完成实机部署。

## 从哪里看起

- [项目进展与结果边界](../PROJECT_STATUS.md)：CAD、训练、头控和硬件的实际阶段。
- [头部 IMU 对照结果](research/20260914_head_attitude/RESULTS.md)：两种策略、仅滤波对照、陈旧数据回退及失败记录。
- [头控对照视频](research/20260914_head_attitude/videos/v7/comparison.mp4) 与 [对照图](research/20260914_head_attitude/comparison.png)。
- [快走和小跳研究](research/20260914_run_jump/RESULTS.md)：速度提升、停止退步、组合小跳和压力条件。
- [快走视频](research/20260914_run_jump/evaluation/run/fast_walk.mp4)、[小跳慢放](research/20260914_run_jump/evaluation/jump_return/jump_return_slow.mp4)。均为仿真。
- [动作达标矩阵](ACTION_STATUS.md)：统一列出已通过、研究候选和未达标项目及其分母。
- [起身两阶段微调复核](research/20260921_action_requalification/RESULTS.md)：碰撞开启、同初态配对的 40+40 条验收。

## CPU 头控复现

先在仓库根目录执行 `git lfs pull`。使用 Python 3.12，新建独立环境：

```bash
python -m venv .venv-sim
# Windows PowerShell
.venv-sim/Scripts/python.exe -m pip install -r simulation/requirements-cpu.txt
.venv-sim/Scripts/python.exe simulation/run_head_demo.py --policy v7 --head-mode imu --seed 1 --seconds 12 --out local_runs/head_imu
.venv-sim/Scripts/python.exe simulation/run_head_demo.py --policy v7 --head-mode filter_only --seed 1 --seconds 12 --out local_runs/head_filter
.venv-sim/Scripts/python.exe simulation/run_head_demo.py --policy v7 --head-mode off --seed 1 --seconds 12 --out local_runs/head_off
.venv-sim/Scripts/python.exe simulation/run_head_tests.py
```

Linux 使用 `.venv-sim/bin/python`。`--policy s42_no_neck` 选择另一候选。每次输出路径必须不存在，脚本会拒绝覆盖旧结果。入口只运行 CPU 仿真，不连接串口、相机或硬件。

`nominal.mjb.gz` 是历史名义物理模型的无损压缩；加载时在内存解压，保持原始模型字节。它依赖 MuJoCo 3.10.0，不保证跨版本二进制兼容。发布版 `contract.json` 使用仓库相对路径。历史模型的碰撞覆盖仍有局限，不能用此次复现宣称完整接触物理已验证。

## 训练源码与模型资产

`training/` 来自 `pollen-robotics/microduck_rl`，基准提交及复制前文件哈希见 [SOURCE_SNAPSHOT.json](SOURCE_SNAPSHOT.json)。快照包含本地未提交修改，不是上游原版或新的原创训练框架；许可为上游 Apache-2.0，另见 [模型来源说明](../NOTICE.md)。

```bash
python simulation/prepare_training_assets.py
cd simulation/training
uv sync
```

资产脚本校验压缩包及每个文件的哈希，只恢复缺失文件，遇到不同内容拒绝覆盖。`training_assets.zip` 包含注册表所需的上游和兼容模型网格；旧模型目录只是源码依赖，不是第二个现用 CAD。当前机械输入始终是仓库 `current/MicroDinosaur_v1.blender`，现用仿真 XML 是 `training/src/mjlab_microduck/robot/microdinosaur_v07/robot_microdinosaur_v07.xml`。

GPU 训练依赖 CUDA/PyTorch、mjlab 1.3.0 和 MuJoCo Warp。`pyproject.toml` 与 `uv.lock` 保留原内容；本机历史训练使用 PyTorch 2.9.1+cu128、Warp 1.12.0、rsl-rl-lib 5.0.1。新机器须先验证 CUDA 设备、运行 64 环境 × 5 迭代冒烟，再启动有预算的训练。本次未在全新 GPU 环境安装依赖或重新训练，不宣称跨机器训练已验收。

`research/train_candidate.py` 是带参数的续训入口，`--checkpoint` 指向用户自行提供的已有检查点，`--xml` 指向当前 v07；`--out` 必须是新的目录。建议配合 `--hardware-profile s288-protocol --reward-recipe v7`。本仓库保存精选推理 ONNX，不包含完整 PPO 检查点池；ONNX 不能代替训练检查点续训。

## 快照与可移植性边界

`research/` 顶层只保留 CPU 头控演示的依赖闭包和通用候选训练入口。依赖未发布输入的临时研究脚本、重复源码快照和本机输出路径已移除。对外验证入口是 `run_head_demo.py`、`run_head_tests.py` 和资产恢复脚本。

为使所选 CPU 演示可复现，`evaluate_policy.py` 支持读取压缩 MJB，`hardware_sim.py` 从本仓库读取总线图，`run_heading_stable_start.py` 使用相对路径读取精选候选。反馈、动力学、奖励、策略和验收阈值未改动。当前发布文件的来源和 SHA 记录在 `SOURCE_SNAPSHOT.json` 与仓库清单中。

## 2026-09-21 整理验证

验证记录见 [PUBLICATION_CHECKS.json](PUBLICATION_CHECKS.json)。历史的 73+9 条头控记录与本次复测分别记录，不混算样本。简历、个人联系方式、环境、完整日志和原始检查点池均不在发布范围。
