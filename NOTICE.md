# 来源与许可

## 机器人几何

MicroDinosaur 使用并修改了 [Pollen Robotics Microduck](https://github.com/pollen-robotics/microduck_rl) 的数字模型，基线提交 `29e887ecfbf5d37144759e5a9f8a176dfb83d547`。当前 OS-M02 继续保留上游头腿等几何，结合本项目头部、骨架、安装和打印适配，不应被描述为全部原创机器人。

上游代码许可为 Apache-2.0；上游 README 对三维模型单列 Creative Commons BY-SA-NC，未指定 CC 版本。模型许可不能用代码许可代替，也不自行补写版本。原始表述见 [模型来源说明](source/microduck_rl/MODEL_LICENSE_NOTE.md)，代码许可见 [LICENSE](source/microduck_rl/LICENSE)。

## HAT 与主机软件

HAT 改编自 Pollen RPI Robot HAT，保留 Apache-2.0 许可、设计出处和修改说明，见 [硬件 NOTICE](hardware/head_hat/NOTICE.md)。S288 主机协议代码独立实现，保留官方协议来源，不包含宇树固件二进制。

## 深度感知

固定 MiDaS RKNN 权重源于 Intel ISL MiDaS Small，保留 [MIT 许可](software/perception/MiDaS_LICENSE)。RKNN SDK、Blender、KiCad、编译器和厂家安装程序不随包分发。相关名称属于各自权利人，不表示官方认证或背书。
