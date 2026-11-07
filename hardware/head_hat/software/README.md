# HEAD B2 — S288 主机适配与 EEPROM

这份代码由主机运行。FT2232H 自带 USB/UART 功能，本板没有运行宇树 STM32 示例的 MCU；不用把该 HAL 工程烧进 HAT，也不因更换转接板而刷新 S288 内部固件。

协议依据为宇树 `digital_servo` 提交 `75c1ed63579dc7bb759e2990bd4b804c11b57daf` 及 J288/S288 官方使用手册。代码独立实现帧编码，修正负角度的有符号编码，提供流式收包、CRC 检查和双端口稳定绑定。现有机器人的零位、方向、关节映射和看门狗继续由上层程序管理。

## Radxa 主机

在目标 Linux 上确认内核有 `ftdi_sio` / `usbserial`。这通常是内核自带驱动，不能把主机串口驱动和舵机固件混为一谈。

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest -v test_s288_ftdi test_eeprom_image
python host_setup.py --serial MDHB20001
```

上面最后一个命令只读取身份和延迟配置，不打开串口、不发电机命令。每块板使用自己的序列号。确认两路均对应此板后，用具备 sysfs 写权限的账号执行：

```sh
sudo .venv/bin/python host_setup.py --serial MDHB20001 --apply
```

它只将这块板的两个 FTDI `latency_timer` 设为 1 ms 并读回。重新插拔/重启后须重新检查。此配置不保证 50 Hz；仍须实测双路并行轮询的最大延迟、丢帧、CRC、断线和负载响应。

上层通过 `bind_ports(inventory_linux(), board_serial)` 找到 A/B，再分别建立 `Bus`，不要假定 `/dev/ttyUSB0` 永远是 A。A、B 是独立锁，每路只允许一个未完成请求。`Bus` 构造不打开设备，只有显式 `open()` 打开串口。通信使用 6,000,000 baud、8N1，无 RTS/CTS；板上 TXDEN 逻辑负责半双工换向，不用软件翻转 RTS。

`command()` 参数为输出轴 SI 单位，ID 为 0–14。15 是广播，当前请求/应答 API 拒绝广播。模式 1 必须启用舵机超时保护。I/O 异常、舵机错误或接收超时会关闭此路，必须由操作者/上层状态机明确复位后重开。它没有接管现有 `servo_workbench`，没有启动任何实物电机服务，也不是整机步态调度器。

## EEPROM

U111 是 93LC56B（16 位组织，256 字节）。配置为 FT2232H、0403:6010、双 UART/VCP、自供电、PWRSAV# / suspend_dbus7、唯一序列号。不开启 remote wakeup。

```sh
python configure_eeprom.py --serial MDHB20001 --output eeprom_plan.json
```

默认仅生成配置计划；本仓库不附带设备 EEPROM 镜像，每块板应使用自己的序列号。离线测试只验证内存中的序列化结果。

组装后先完成限流上电、3.3 V/1.8 V/5 V 检查和 USB 枚举，断开舵机动力。明确选中此板的 USB URL，才执行：

```sh
python configure_eeprom.py --serial YOUR_UNIQUE_SERIAL --url ftdi://ftdi:2232h:CURRENT_SERIAL/1 --output board_eeprom.json --apply
```

`CURRENT_SERIAL` 指烧写前设备实际报告的序列号；空白/同号设备不能照抄示例，应只接一块并根据 PyFtdi 枚举结果选择准确 URL。工具先保存原始 256 字节，再生成新镜像，写入后由 PyFtdi 逐字节回读比较。断电重启后再核对两路 UART、序列号，以及 USB 未配置、挂起和主板掉电时两个发送使能均释放。

EEPROM 工具固定 PyFtdi 0.57.1，并测试其实际序列化 API、USB bmAttributes 必须为 0xC0、存储器类型 0x56、校验和和解码字段。Linux 下 USB 写入需要适当权限；Windows 若当前使用 FTDI VCP 驱动，优先使用官方 FT_PROG 按同一参数配置，不要为运行 PyFtdi 盲目替换现有 USB 驱动。

## 交付边界

11 项离线测试已通过；未对实物写 EEPROM，未在 Radxa 上部署，未发送 S288 指令。音频 codec 的设备树/ALSA、SPI IMU 和 ToF 的板端驱动需结合目标 Radxa 内核、设备树与实际模块验证；本目录不冒充整机系统镜像。
