# 当前模型与打印

本目录只保留 OS-M02 最新整机和匹配主打印工程。打开前可运行仓库根目录的 `python scripts/verify_current.py`。

- [MicroDinosaur_Current.blend](MicroDinosaur_Current.blend) 为 2026-11-08 保存的 `edit_20261108_220551`，包含更新的头壳侧槽和 M2 导孔。
- [MicroDinosaur_Print.3mf](MicroDinosaur_Print.3mf) 有 8 盘 30 件，逐盘件数为 `[10,5,8,2,1,1,2,1]`，第 2 盘同步更新的头壳。
- [current_manifest.json](current_manifest.json) 记录文件哈希、对象和驱动数及验证边界。
- [头壳验证](validation/current_head_vent_position_m2.json) 对应当前 CAD 和主打印工程。[四孔复核](validation/current_opposed_mount_holes.json) 记录沿用的 11 月 2 日 yaw2roll 支架检查。

详细使用边界见 [当前设计](../CURRENT_DESIGN.md)。

## 整机零位校准架

[三件夹具与两盘工程](calibration_fixture/README.md)包含夹具 Blender 建模源、两盘打印工程、整机示意图及最终检查。装配参考和本目录的主机器人 CAD 采用同一 11 月 8 日整机源文件。
