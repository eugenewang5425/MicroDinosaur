# 当前机械设计

发布核对：2026-11-07。阶段 **OS-M02**，几何状态 `edit_20261102_090559`。

唯一整机入口为 [MicroDinosaur_Current.blend](current/MicroDinosaur_Current.blend)，配套 [MicroDinosaur_Print.3mf](current/MicroDinosaur_Print.3mf)。文件名保持固定，小尺寸、公差和排盘修改不升级整机阶段。文件身份见 [current_manifest.json](current/current_manifest.json)。

- 150 对象、124 网格、6 条保留驱动；19 个精细舵机实例属于参照，不是打印件，也不表示全部关节都已驱动。
- 主工程 8 盘 30 件，逐盘件数为 `[10,5,8,2,1,1,2,1]`；脸板第 1 盘、相机盖第 2 盘、头底座第 5 盘、电池座第 6 盘、TPU 牙床第 7 盘、骨架第 8 盘。
- 当前最新改动为两件 yaw2roll 相向圆耳共四个安装导孔 Ø2.1→Ø1.8 mm，孔轴和深度不变。已在保存的 CAD 与原生 3MF 中逐孔复核，第 1 盘原生切片无警告；其余 28 个打印网格及原工艺、材料和摆放保留。
- 相机四柱名义孔距 21×13.5 mm，柱高 6.3 mm，Ø1.7×5.2 mm 盲导孔；13.5 mm 由照片推算，尚需实物核对。相机盖采用中央贯通、三卡扣方案，须与当前脸板配对使用。
- 两腿大轴承凸台为 Ø15.9→Ø16.1→Ø16.2 mm 的连续单调曲面，其余大凸台保持既有尺寸；不追加全局打印补偿。

最新检查见 [四孔结果](current/validation/current_opposed_mount_holes.json)、[保存网格测量](current/validation/saved_validation.json)、[打印工程检查](current/validation/project_validation.json) 和 [切片结果](current/validation/slice_validation.json)。这些是已有检查的整理，本次没有重新建模、切片或下发打印。

实物配合、螺丝咬合、承载、线束避让、完整连续运动和制造验收尚未完成。原结构中的已知局部落座接触保留。当前发布包没有可直接用于该 OS-M02 的已验收全身质量/惯量台账或整机步态策略。
