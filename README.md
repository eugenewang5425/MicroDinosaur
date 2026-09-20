# MicroDinosaur

小型双足恐龙机器人的机械改型、强化学习训练与 IMU 控制实验。**2026-09-20 更新**：已整理当前训练代码、所选 ONNX、CPU 头控复现入口及精选对照结果。项目处于设计与仿真到实物搭建的过渡阶段，尚未完成机身打印与实机验证。

先看 [项目进展](PROJECT_STATUS.md)；代码、演示与复现入口见 [仿真与训练](simulation/README.md)。CAD 版本仍为下述 v07，文件哈希未改变。

MicroDinosaur v1 当前机械 CAD，版本为 **2026-09-13 / 头部 JY61P v07**。唯一现用模型是 [current/MicroDinosaur_v1.blender](current/MicroDinosaur_v1.blender)：755 对象、663 网格、672 条实体质量台账、19 个 S288 / 19 个驱动，头部和躯干各一块名义 JY61P。

![当前整机](current/previews/assembly_three_quarter.png)

先读 [当前设计](CURRENT_DESIGN.md)，再读 [v07 头部 IMU 评审](HEAD_IMU_V07_REVIEW.md)。模型 SHA256：

```text
0a4f86aefea24e0d5260c837849a16164ec83cc2d364f58e6c03947de95ce286
```

## 获取与打开

安装 Git LFS 与 Blender 4.5 LTS；本项目验证环境为 Blender 4.5.9。模型使用 Git LFS 保存，下载普通 ZIP 可能只得到指针文件，推荐克隆：

```bash
git lfs install
git clone https://github.com/eugenewang5425/MicroDinosaur.git
cd MicroDinosaur
git lfs pull
python scripts/verify_current.py
```

Windows 双击 `OPEN_MICRODINOSAUR_CURRENT.bat`。启动器依次使用 `BLENDER_EXE` 指定的程序、本地 `tools/blender-4.5.9-windows-x64/blender.exe`、PATH 中的 `blender.exe`。Blender 安装包不随仓库上传。也可以直接运行：

```bash
blender current/MicroDinosaur_v1.blender --python blender_controls.py
```

`.blender` 是用户指定的项目后缀，已验证命令行可以直接打开；无需改名。

## 文件组织

| 路径 | 内容 |
| --- | --- |
| `current/` | 唯一现用模型、清单、质量、紧固接口、IMU 外参、总线与 6 张当前预览 |
| `scripts/verify_current.py` | 无旧版本依赖的文件/台账校验；在 Blender 中还可检查实际对象、驱动和 v07 网格 |
| `scripts/head_imu_v07/` | 最新增量建模、静态/运动审计、渲染和发布脚本；详细依赖见该目录 README |
| `blender_controls.py` | 当前入口使用的 Blender 关节审阅面板 |
| `work_in_progress/head_imu_v07/` | 仅上传 v07 的增量几何和最终检查记录，保留原证据路径及字节 |
| `source/microduck_rl/` | 与当前模型来源相关的许可及说明 |
| `repository_files.json` | 本次上传文件逐项清单、大小和 SHA256（不包含清单自身） |
| `simulation/` | 当前训练源码快照、压缩资产、精选策略、头控 CPU 入口与历史实验结果 |
| `PROJECT_STATUS.md` | 设计、仿真与硬件的当前阶段及结果边界 |

Git 采用明确的文件白名单。旧 CAD 总成、采购资料、Blender 安装包、虚拟环境、缓存、个人简历及完整训练日志/检查点池留在本地。`simulation/` 仅收录明确选定的源码、必需兼容资产、策略和结果；它不增加第二个现用机械入口。

原设计文档和 JSON 中的历史路径是来源/恢复记录，可能指向未上传的本地文件；这些记录保留原样，不是克隆后的必要运行路径。远程仓库可打开、检查和渲染当前模型，但没有从旧版本完整重放增量构建的输入。下一轮建模应先冻结当前版本，再开展新的增量修改，见 [脚本说明](scripts/head_imu_v07/README.md)。

## 校验与边界

`python scripts/verify_current.py` 仅需 Python 标准库；更深入的当前 CAD 检查运行：

```bash
blender --background current/MicroDinosaur_v1.blender --python-exit-code 1 --python scripts/verify_current.py
```

这些命令不保存或改动模型。已存档的 1605 姿态结果来自 v07 发布时的离散审计，便携校验只核对其绑定关系，不冒称重新执行该历史姿态集。

当前模型尚未获得制造、打印、强度、电气、实机、连续碰撞、步态或有效 COM 放行。质量约 1098.214 g 为未称重估算；真实增强款 IMU、接头、线束和螺钉头需试装，头部最近名义螺钉间隙约 0.616 mm。具体限制以当前设计与评审为准。

第三方几何来源与许可见 [NOTICE.md](NOTICE.md)。
