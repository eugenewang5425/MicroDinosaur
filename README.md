# MicroDinosaur

基于 Microduck 改型的小型双足恐龙机器人，使用 S288 舵机、Radxa ZERO 3W、Blender 和 Python。当前工作包括机械与打印适配、板载深度感知、IMU 接入和双总线电控设计。开发中使用 AI 辅助；来源与许可见 [NOTICE](NOTICE.md)。

**更新至 2026-11-24。** 项目处于研究原型阶段。整机试装、承载、实机站立和行走仍待验证。

| 内容 | 当前入口 | 状态 |
| --- | --- | --- |
| 整机 CAD | [MicroDinosaur_Current.blend](current/MicroDinosaur_Current.blend) | OS-M02，11 月 8 日头壳更新，150 对象、124 网格、6 条保留驱动 |
| 打印工程 | [MicroDinosaur_Print.3mf](current/MicroDinosaur_Print.3mf) | 8 盘 30 件，第 2 盘头壳与当前 CAD 配套 |
| 整机零位校准架 | [三件夹具、两盘工程与示意图](current/calibration_fixture/README.md) | 11月8日最新试装版；两盘切片通过，实物配合与承载待验 |
| 头部集成板 | [KiCad 工程与说明](hardware/head_hat/README.md) | 本仓库附 B2 V3 审阅包。R57 四层与六层候选的数字复核见项目状态 |
| S288 主机接口 | [双总线适配代码](hardware/head_hat/software/README.md) | 帧编码、CRC、端口绑定与 EEPROM 工具；待实板联调 |
| IMU | [接线、读取程序与数据](hardware/imu/README.md) | F411 与 ZERO 3W 基础读取通过；异常跳点待排查 |
| 板载感知 | [当前源码、模型与复测边界](software/perception/README.md) | 相机/NPU 功能已恢复，散热后持续性能待复测 |
| 步态仿真 | [Demo 1 领域随机化](https://github.com/eugenewang5425/MicroDinosaur/tree/research/demo1-domain-randomization/research/demo1_domain_randomization) | 19 轴研究模型，四秒前进通过，归零后十秒停稳未通过 |

完整阶段和验收证据见 [项目状态](PROJECT_STATUS.md)，机械文件说明见 [当前设计](CURRENT_DESIGN.md)。

## 研究分支

默认分支用于 OS-M02 发布。历史校准的负结果保留在归档分支，正在验证的步态方案保留在研究分支。详见 [六轴校准归档](https://github.com/eugenewang5425/MicroDinosaur/blob/archive/s288-calibration-20261124/research/s288_calibration_20261124.md) · [研究索引与分支约定](https://github.com/eugenewang5425/MicroDinosaur/tree/research/demo1-domain-randomization/research)。

## 获取与检查

```sh
git lfs install
git clone https://github.com/eugenewang5425/MicroDinosaur.git
cd MicroDinosaur
git lfs pull
python scripts/verify_current.py
```

CAD 和 RKNN 权重通过 Git LFS 保存。直接下载 ZIP 时，需确认大文件不是 LFS 指针。用 Blender 4.5 打开 `current/MicroDinosaur_Current.blend`，用 Bambu Studio 打开原生 `current/MicroDinosaur_Print.3mf`；先核对打印机、材料和工艺。读取或打开文件不会下发打印任务。

OS-M02 的 CAD 含 19 个舵机参照实例及 6 条现有驱动。Demo 1 仿真使用独立的 19 轴动力学模型与执行合同。

仓库不保存中间整机、`.blend1`、前后重复副本、过程截图、失败候选、完整日志、工具安装包、个人设备配置和实时标定记录。各模块只附与本次交付有关的最终检查或测量结果。
