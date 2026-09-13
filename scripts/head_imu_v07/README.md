# v07 建模脚本

本目录保存产生当前头部 JY61P 安装结构的最新增量脚本。整理时只调整脚本路径、抽出参考姿态函数，并使渲染能独立读取当前模型；几何参数和当前 CAD 文件未修改。

## 克隆后可直接使用

- `../../blender_controls.py`：当前模型的关节审阅面板，由根目录 BAT 加载。
- `../verify_current.py`：文件哈希、台账、预览和 v07 证据一致性检查；在 Blender 中还核对当前对象/父子关系、有效驱动、质量、IMU 外参、9 个增改网格和外部资源。
- `head_imu_v07_blender.py -- render`：从当前模型生成 6 张预览，不保存模型；默认输出到未跟踪的 `renders/head_imu_v07/`，已有同名图片时拒绝覆盖。可用 `--output` 指定新输出目录。

在仓库根目录执行：

```bash
python scripts/verify_current.py
blender --background current/MicroDinosaur_v1.blender --python-exit-code 1 --python scripts/verify_current.py
blender --background current/MicroDinosaur_v1.blender --python-exit-code 1 --python scripts/head_imu_v07/head_imu_v07_blender.py -- render
```

查看、校验和渲染不需要安装 `requirements-modeling.txt`。Blender 4.5.9 自带 Python、NumPy 与 `bpy`。

## 保留的 v07 增量配方

| 脚本 | 用途 | 原始输入依赖 |
| --- | --- | --- |
| `head_imu_v07_snapshot.py` | 冻结修改前模型、属性、网格及局部头部库 | 原 v06 模型，SHA256 必须为 `656665835b6ea6c170a2648cca2630ce6dbc186811fbceea633f3260b9ebceb4` |
| `head_imu_v07_geometry.py` | 托座、压条、板卡及五金名义几何 | `basis.json` 和 `basis_meshes.npz`；Python NumPy / manifold3d |
| `head_imu_v07_check.py` | 静态交集、螺母装入及工具通道 | 同上及 `delta.json` / `delta_meshes.npz` |
| `head_imu_v07_motion.py` | 增量运动审计和板卡拆卸路径 | 同上及本地 `work_in_progress/lower_module_v05/` 冻结姿态数组 |
| `head_imu_v07_blender.py` 的 `build` / `verify` | 候选集成、重开和历史姿态回放 | 原冻结基线、源质量记录、旧姿态数组、原 Microduck 校验文件 |
| `head_imu_v07_scene_metadata.py` | 当前 Scene 标签清理函数及一次性迁移配方 | 独立执行仅接受原清理前哈希；不要在当前版本重复运行 |
| `publish_head_imu_v07.py` | 一次性本地发布、保存恢复库及截图记录 | 原冻结输入、候选、完整发布检查和本地历史目录 |
| `reference_pose.py` | 恢复对象自身记录的参考角 | 当前打开的 Blender 场景 |

这些配方输入包含旧版本，按本次上传范围留在本地；GitHub 中仅保留 v07 最终 `delta` 和检查记录。缺少基线时生成/发布入口会停止，不能使用当前 v07 代替旧输入。要完整重放这次增量流程，必须在原本地档案环境中使用对应冻结数据。下一版应另建冻结目录和新版本脚本，不能叠加旧生成器。

普通 Python 几何依赖记录在根目录 `requirements-modeling.txt`。当前的 `requirements-lock.txt` 属于本地更广的历史工作环境，未纳入本次上传。

`work_in_progress/head_imu_v07/*_checks.json` 是发布时证据；独立检查不重写这些文件。报告中的旧绝对路径保留来源语义，当前便携脚本使用仓库相对路径。
