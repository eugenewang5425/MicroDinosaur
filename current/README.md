# 当前模型与打印

本目录只保留 OS-M02 最新整机和匹配主打印工程。打开前可运行仓库根目录的 `python scripts/verify_current.py`。

- `MicroDinosaur_Current.blend`：2026-11-02 保存状态 `edit_20261102_090559`。
- `MicroDinosaur_Print.3mf`：8盘30件 `[10,5,8,2,1,1,2,1]`。
- `current_manifest.json`：哈希、对象/驱动数及验证边界。
- `validation/`：仅本版四孔修正对应的最终测量、打印和切片记录。

详细使用边界见 [当前设计](../CURRENT_DESIGN.md)。本次发布没有改动模型与打印文件的字节。

## 整机零位校准架

[最新三件夹具与两盘工程](calibration_fixture/README.md)包含带机器人参考的Blender/3MF装配及最终检查。参考头壳已同步11月8日侧槽和M2导孔；本目录主机器人CAD/八盘打印文件保持11月2日的原发布字节，来源状态在零位架清单内单独标注。
