"""Local Blender panel for this project; loaded by the included launchers.
No networking and no persistent Blender installation changes.
"""
import bpy
from mathutils import Vector

SHELLS={'top_head_shell','left_shell','right_shell','upper_leg_left','upper_leg_right','face_part','noenoeil','soft_mouth_top','jaw_soft',
        'head','body_front','body_back','body_middle_top','body_middle_bottom','battery_pack_lid','head_bot_sheet','left_cache','right_cache'}
NAMES={'left_hip_yaw':'左髋偏航','left_hip_roll':'左髋侧倾','left_hip_pitch':'左髋俯仰','left_knee':'左膝','left_ankle':'左踝',
       'right_hip_yaw':'右髋偏航','right_hip_roll':'右髋侧倾','right_hip_pitch':'右髋俯仰','right_knee':'右膝','right_ankle':'右踝',
       'neck_pitch':'颈部俯仰','head_pitch':'头部俯仰','head_yaw':'头部偏航','head_roll':'头部侧倾','left_antenna':'左触角','right_antenna':'右触角'}

def controls():return [o for o in bpy.data.objects if o.name.startswith('CTRL_') and 'angle_deg' in o]
def dino_controls():return [o for o in bpy.data.objects if o.name.startswith('DCTL_') and 'angle_deg' in o]
def toggle_shells(self,context):
    if context.scene.get('s288_integrated'):
        for name in ['Rex_Skull_Shell','V11_Open_Side_Guard_L','V11_Open_Side_Guard_R']:
            obj=bpy.data.objects.get(name)
            if obj:obj.hide_set(context.scene.duck_cutaway);obj.hide_render=context.scene.duck_cutaway
        return
    for obj in bpy.data.objects:
        if obj.get('v10_retired') or obj.get('v11_retired') or obj.get('v12_retired') or obj.get('v13_retired'):continue
        if obj.get('v13_new'):
            if obj.name=='Rex_Skull_Shell':
                obj.hide_set(context.scene.duck_cutaway);obj.hide_render=context.scene.duck_cutaway
            continue
        if obj.get('v12_new'):continue
        if obj.get('v11_new'):
            if obj.name.startswith('V11_Open_Side_Guard'):
                obj.hide_set(context.scene.duck_cutaway);obj.hide_render=context.scene.duck_cutaway
            continue
        if obj.get('v10_new'):
            if obj.name in {'V10_Shell_Wrap','V10_Back_Service_Cover','V10_Battery_Service_Hatch'}:
                obj.hide_set(context.scene.duck_cutaway);obj.hide_render=context.scene.duck_cutaway
            continue
        if str(obj.get('duckrex_role','')).startswith('hidden_'):continue
        if obj.name=='Rex_Tail_Chassis_Bridge' or obj.name.startswith(('Rex_Chassis','Rex_Yaw','Rex_Pitch','Rex_Rigid','Rex_Arm_Clamp','Rex_OpenRB','Rex_Camera_Pi')):continue
        if obj.get('source_mesh') in SHELLS or obj.get('print_part'):
            obj.hide_set(context.scene.duck_cutaway);obj.hide_render=context.scene.duck_cutaway
def toggle_study(self,context):
    col=bpy.data.collections.get('08_STUDY_MARKERS')
    if col:
        if context.scene.get('v12_rear_hips'):
            col.hide_viewport=True;col.hide_render=True
            return
        col.hide_viewport=not context.scene.duck_study;col.hide_render=not context.scene.duck_study
def toggle_rig(self,context):
    for obj in controls():obj.hide_set(not context.scene.duck_show_axes)

def toggle_reservations(self,context):
    col=bpy.data.collections.get('90_DESIGN_RESERVATIONS_NOT_PARTS')
    if col:
        for obj in col.objects:obj.hide_set(not context.scene.duck_show_reservations)

class DUCK_OT_reference(bpy.types.Operator):
    bl_idname='duck.reference';bl_label='恢复参考姿态';bl_description='恢复源 MJCF 参考姿态；不代表机器人已经能够平衡'
    def execute(self,context):
        context.scene.frame_set(1)
        for obj in controls()+dino_controls():obj['angle_deg']=obj['reference_deg'];obj.update_tag()
        for obj in bpy.data.objects:
            if 'passive_angle_deg' in obj:obj['passive_angle_deg']=0.;obj.update_tag()
        context.view_layer.update();return {'FINISHED'}

class DUCK_OT_com(bpy.types.Operator):
    bl_idname='duck.com';bl_label='计算当前重心';bl_description='原连杆质量加有质量属性的打印件/新增硬件估计；参考支撑轮廓不能判定动态稳定'
    def execute(self,context):
        if context.scene.get('clean_current_assembly'):
            self.report({'WARNING'},'当前审阅版电池、功率模块与质量惯量未定型，不能输出有效重心或沿用旧步态。')
            return {'CANCELLED'}
        if context.scene.get('v12_rear_hips'):
            self.report({'WARNING'},'v0.12 髋腿与电池已迁移，质量惯量和步态尚未重建，旧重心与支撑面无效。')
            return {'CANCELLED'}
        if context.scene.get('v09_interfaces_closed'):
            self.report({'WARNING'},'v0.9 尚未完成源连杆质量去重和板卡称重，暂不输出有效重心。')
            return {'CANCELLED'}
        context.view_layer.update();weighted=Vector((0,0,0));total=0
        for obj in bpy.data.objects:
            if 'mass_kg' in obj and 'com_local_m' in obj:
                mass=obj['mass_kg'];weighted+=mass*(obj.matrix_world@Vector(obj['com_local_m']));total+=mass
        if total:
            point=weighted/total;context.scene.duck_com_text=f'{point.x*1000:.1f}, {point.y*1000:.1f}, {point.z*1000:.1f} mm'
            marker=bpy.data.objects.get('REFERENCE_COM_FROM_MJCF_MASSES')
            if marker:marker.location=point
            curve=bpy.data.objects.get('COM_gravity_projection')
            if curve:
                curve.data.splines[0].points[0].co=(*point,1)
                curve.data.splines[0].points[1].co=(point.x,point.y,0,1)
            self.report({'INFO'},f'Mass {total:.3f} kg; CoM {context.scene.duck_com_text}')
        return {'FINISHED'}

class DUCK_PT_workbench(bpy.types.Panel):
    bl_label='Duck 装配与关节';bl_idname='DUCK_PT_workbench';bl_space_type='VIEW_3D';bl_region_type='UI';bl_category='Duck'
    def draw(self,context):
        layout=self.layout;scene=context.scene
        layout.label(text='数字装配基线 / 单位：毫米',icon='MOD_ARMATURE')
        layout.label(text='关节范围来自仿真，未做实机防碰撞。')
        if scene.get('s288_integrated'):
            layout.label(text='19 × 宇树 S288 / 已集成换型审阅版')
            layout.label(text='3S 候选；电气、运动及强度未放行',icon='ERROR')
            layout.label(text='腿部极限仍碰撞；以下源范围不可上机',icon='ERROR')
        elif scene.get('clean_current_assembly'):
            layout.label(text='v0.13 当前干净装配：无隐藏旧零件')
            layout.label(text='电池/电源待定型；未通过实机验收',icon='ERROR')
        elif scene.get('v12_rear_hips'):
            layout.label(text='前置电池 / 后移双腿：完整髋部总成已迁移')
            layout.label(text='重心与步态需重建；未通过实机验收',icon='ERROR')
        elif scene.get('v11_open_service'):
            layout.label(text='开放检修版：原版腿护件与承力骨架保留')
            layout.label(text='电池、电源模块及实机装配未验收',icon='ERROR')
        elif scene.get('v10_battery'):
            layout.label(text='6 V 布局审阅：电池/电源模块尚未选型',icon='ERROR')
            layout.label(text='大幅后倾+侧转仍有背盖干涉；勿上电执行')
        layout.operator('duck.reference',icon='LOOP_BACK')
        layout.prop(scene,'duck_cutaway',text='隐藏覆盖外壳，查看硬件')
        row=layout.row();row.enabled=not scene.get('v12_rear_hips')
        row.prop(scene,'duck_study',text='显示参考支撑面与重心')
        layout.prop(scene,'duck_show_axes',text='显示关节控制环')
        if scene.get('clean_current_assembly'):
            layout.prop(scene,'duck_show_reservations',text='显示空间预留（非零件）')
        layout.separator()
        for group,title in [('left_','左腿 / 左触角'),('right_','右腿 / 右触角'),('head','头部 / 颈部')]:
            box=layout.box();box.label(text=title.split(' / ')[0] if scene.get('clean_current_assembly') and group!='head' else title)
            for obj in controls():
                name=obj.name.removeprefix('CTRL_')
                if (name.startswith(group) if group!='head' else name.startswith(('head','neck'))):
                    box.prop(obj,'["angle_deg"]',text=NAMES.get(name,name),slider=True)
        if dino_controls():
            box=layout.box();box.label(text='DuckRex 附加机构')
            for obj in dino_controls():
                powered = bpy.data.objects.get('TAIL_XL330_M288_T') is not None
                label=('尾根俯仰（XL330 主动）' if powered else '尾巴弯曲（五节联动/预留）') if 'Tail' in obj.name else '下颌开合（展示/预留）'
                if obj.name=='DCTL_Tail_Yaw':label='尾巴左右（SC09，±25°）'
                if obj.name=='DCTL_Tail_Pitch':label='尾巴上下（SC09，±20°）'
                if obj.name=='DCTL_Jaw_Hinge' and not obj.get('display_only',True):
                    label='嘴部开合（匹配原 XL330；数字预览）'
                    if scene.get('v09_preview_interlock') and bpy.data.objects['CTRL_neck_pitch']['angle_deg']<20:
                        box.label(text='大幅前倾时预览强制闭嘴；非硬件保护',icon='ERROR')
                if obj.name=='DCTL_Arm_L':label='左小臂（SC09）'
                if obj.name=='DCTL_Arm_R':label='右小臂（SC09）'
                if scene.get('s288_integrated'):
                    label={'DCTL_Tail_Yaw':'尾巴左右（S288）','DCTL_Tail_Pitch':'尾巴上下（S288）','DCTL_Jaw_Hinge':'嘴部开合（S288）','DCTL_Arm_L':'左小臂（S288）','DCTL_Arm_R':'右小臂（S288）'}.get(obj.name,label)
                box.prop(obj,'["angle_deg"]',text=label,slider=True)
            passive=[o for o in bpy.data.objects if o.name.startswith('RIG_Tail_Hinge_') and 'passive_angle_deg' in o]
            if passive:
                box.label(text='后四节：被动摩擦关节，以下仅预览')
                for obj in sorted(passive,key=lambda o:o.name):
                    box.prop(obj,'["passive_angle_deg"]',text=f'尾节 {obj.name[-2:]} 被动角',slider=True)
        layout.operator('duck.com',icon='PIVOT_CURSOR')
        layout.label(text=scene.duck_com_text)
        if scene.get('clean_current_assembly'):
            layout.label(text='失效的旧重心/支撑面已移除。')
        else:
            layout.label(text='绿色轮廓仅适用于参考站姿。')
            layout.label(text='改腿部姿态后需重新计算接触。')

classes=(DUCK_OT_reference,DUCK_OT_com,DUCK_PT_workbench)
for cls in classes:
    old=getattr(bpy.types,cls.__name__,None)
    if old:
        try:bpy.utils.unregister_class(old)
        except RuntimeError:pass
    bpy.utils.register_class(cls)
bpy.types.Scene.duck_cutaway=bpy.props.BoolProperty(default=False,update=toggle_shells)
bpy.types.Scene.duck_show_reservations=bpy.props.BoolProperty(default=False,update=toggle_reservations)
bpy.types.Scene.duck_study=bpy.props.BoolProperty(default=False,update=toggle_study)
bpy.types.Scene.duck_show_axes=bpy.props.BoolProperty(default=False,update=toggle_rig)
bpy.types.Scene.duck_com_text=bpy.props.StringProperty(default='点击上方计算重心')
study=bpy.data.collections.get('08_STUDY_MARKERS')
if study:study.hide_viewport=True
for area in bpy.context.screen.areas:
    if area.type=='VIEW_3D':area.spaces.active.show_region_ui=True
print('Duck workbench registered:',len(controls())+len(dino_controls()),'joint sliders (main + added mechanisms)')
