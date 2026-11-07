# MicroDinosaur

基于 Microduck 改型的小型双足恐龙机器人，使用 S288 舵机、Radxa ZERO 3W、Blender 和 Python。当前工作包括机械与打印适配、板载深度感知、IMU 接入和双总线电控设计。开发中使用 AI 辅助；来源与许可见 [NOTICE](NOTICE.md)。

**更新至 2026-11-07。默认分支只保存最新交付文件，旧版本可从 Git 提交历史查看。** 当前仍为研究原型，实机站立、行走、完整装配和制造验证尚未完成。

| 内容 | 当前入口 | 状态 |
| --- | --- | --- |
| 整机 CAD | [MicroDinosaur_Current.blend](current/MicroDinosaur_Current.blend) | OS-M02，11 月 2 日更新；150 对象、124 网格、6 条保留驱动 |
| 打印工程 | [MicroDinosaur_Print.3mf](current/MicroDinosaur_Print.3mf) | 8 盘 30 件，与当前模型配套 |
| 头部集成板 | [KiCad 工程与说明](hardware/head_hat/README.md) | 65×30 mm 六层；原生 DRC 0 错误、0 警告、0 未连接；未打样验收 |
| S288 主机接口 | [双总线适配代码](hardware/head_hat/software/README.md) | 帧编码、CRC、端口绑定与 EEPROM 工具；待实板联调 |
| IMU | [接线、读取程序与数据](hardware/imu/README.md) | F411 与 ZERO 3W 基础读取通过；异常跳点待排查 |
| 板载感知 | [当前源码、模型与复测边界](software/perception/README.md) | 相机/NPU 功能已恢复；散热后持续性能待复测 |

完整阶段和下一步见 [项目状态](PROJECT_STATUS.md)，机械使用边界见 [当前设计](CURRENT_DESIGN.md)。

## 获取与检查

```sh
git lfs install
git clone https://github.com/eugenewang5425/MicroDinosaur.git
cd MicroDinosaur
git lfs pull
python scripts/verify_current.py
```

CAD 和 RKNN 权重通过 Git LFS 保存。直接下载 ZIP 时，需确认大文件不是 LFS 指针。用 Blender 4.5 打开 `current/MicroDinosaur_Current.blend`，用 Bambu Studio 打开原生 `current/MicroDinosaur_Print.3mf`；先核对打印机、材料和工艺。读取或打开文件不会下发打印任务。

当前模型保留 19 个舵机参照实例，但只有 6 条现有驱动；不能据此宣称完整 19 轴运动链已经绑定。旧 v07 的质量台账、19 驱动和仿真验收不适用于当前 OS-M02。

仓库不保存中间整机、`.blend1`、前后重复副本、过程截图、失败候选、完整日志、工具安装包、个人设备配置和实时标定记录。各模块只附与本次交付有关的最终检查或测量结果。
