# 来源

本设计改编自 Pollen Robotics [RPI Robot HAT](https://github.com/pollen-robotics/elec_RPI_Robot_HAT/tree/6de86283297e4c49a9da8312cd3395ab83c437e3)，固定提交 `6de86283297e4c49a9da8312cd3395ab83c437e3`。上游 Apache-2.0 许可见本目录 LICENSE。

本项目修改包括 65×30 mm 六层布局、FT2232H 双 UART/S288 半双工接口、电源保护、传感器接口和音频外围。当前副本为 2026-11-07 的 B2 V3 走线优化结果；板文件仍沿用 HEAD_A0 的工程内名称。板载时钟为有源 MEMS 时钟。

S288 主机适配代码独立实现，协议参考宇树 [digital_servo](https://github.com/unitreerobotics/digital_servo/tree/75c1ed63579dc7bb759e2990bd4b804c11b57daf)。符号和封装保留原文件中的出处。此板不是 Pollen、Radxa 或宇树官方产品。
