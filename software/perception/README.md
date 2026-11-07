# 当前板载深度感知

Radxa ZERO 3W / RK3566，IMX219，MiDaS Small 的固定 INT8 RKNN 模型。发布包含当前相机服务、最新帧流水线、CPU 数值路径、模型专用原生 INT8 输入转换及离线回归测试；只附当前使用的一份权重。

固定条件为相机 640×360、传感器 1920×1080、模型 256×256、JPEG82。权重 SHA256 为 `8a22b419775cae587b5478df04f01ccc7eaf9f4a1585b705fc98781841dd0ac1`。模型与转换代码身份、自检不匹配时拒绝启用 `native-exact`，可回退 SDK 通路。

## 实测状态

既有 18 张真实图、196608 个边界测试输入字节及线程缓冲检查通过，原生深度/JPEG 与 SDK 参考逐字节一致；配对净省约 10.72 ms/帧。单线程 1104 MHz 裸板试验运行 165.71 秒，网页约 19.95 fps，SoC 达 85°C 后停止，**未通过完整 300 秒持续目标**。软件延迟是估计值，不是光子到屏幕实测。最终数据见 [性能摘要](validation/performance.json)、[输入边界](validation/boundary_verification_v2.json) 和 [单线程对照](validation/single_thread_equivalence.json)。

截至 11 月 6 日，排针焊接修复后相机、推理、Wi-Fi、蓝牙功能复测正常。本周没有新增持续性能结论；散热配件尚未购置，安装后再复测。

## 运行和离线验证

普通电脑可运行无相机/NPU的离线测试：

```sh
python -m pip install -r requirements.txt
python -m unittest discover -p 'test_*.py' -v
```

真实板需另外准备匹配的 RKNN Lite/运行时、GStreamer Python 绑定、IMX219 驱动与既有相机配置；这些板级环境不随仓库分发。先拉取 Git LFS 权重。在该目录、已配置好的 ARM64 板上：

```sh
python3 input_transport_20261029/build_exact_input.py
python3 camera_live_server.py --model midas-optimized-int8 --input-transport native-exact --cv-threads 1 --inference-workers 1
```

打开 `http://127.0.0.1:8790/`。构建程序需 gcc，生成本机 `.so` 和身份清单；原生通路在每次上下文创建时对照 SDK 数值。默认省略 `--input-transport` 时使用 SDK。服务启动会核验并按默认参数恢复既有 1080p 传感器模式；需保留当前模式时使用 `--sensor-mode current`。本次发布只执行离线检查，没有启动板端服务或改变板上配置。

MiDaS 的上游来源为 [isl-org/MiDaS](https://github.com/isl-org/MiDaS)，其 MIT 许可保留于 [MiDaS_LICENSE](MiDaS_LICENSE)。本地进行了固定 RKNN 转换和输入搬运适配；仅该固定模型/SDK组合有现有保真证据，不能推广到其它权重。
