"""Display-only step colors, edge markers and a measured-position diagram."""
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


WIDTH, SCENE_HEIGHT, HEADER, HEIGHT = 960, 560, 80, 816
COLORS = ('#38bddd', '#8dca60', '#ca93df')
EDGES = (.30, .48, .66)
FONT = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 21)
SMALL = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 18)


def mark_scene(renderer, model):
    """Change mjvScene drawing only; no simulation model/data is mutated."""
    scene = renderer.scene
    ids = {model.geom(f'step_{i}').id: i for i in range(3)}
    for geom in scene.geoms[:scene.ngeom]:
        if geom.objtype == mujoco.mjtObj.mjOBJ_GEOM and geom.objid in ids:
            color = COLORS[ids[geom.objid]]
            geom.rgba[:] = [int(color[k:k+2], 16)/255 for k in (1, 3, 5)] + [1.]
    for i, x in enumerate(EDGES):
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_BOX,
            np.array([.002, 3., .0006]), np.array([x+.001, 0., (i+1)*.01+.0008]),
            np.eye(3).ravel(), np.array([1., .88, .12, 1.]))
        geom.category = mujoco.mjtCatBit.mjCAT_DECOR
        scene.ngeom += 1


def frame(scene, label, scene_kind, t, base_x, vx):
    output = Image.new('RGB', (WIDTH, HEIGHT), '#111a28')
    output.paste(Image.fromarray(scene), (0, HEADER))
    draw = ImageDraw.Draw(output)
    names = {'legacy': '旧教师约束', 'archive': '历史示范',
             'demonstration': '专项示范训练后的学生', 'recovery': '学生 → 专家接管',
             'expert': '专家示范'}
    title = names.get(label, label)
    draw.text((20, 10), f'{title}  |  三级台阶，每级升高 10 mm', font=FONT, fill='white')
    if scene_kind == 'recovery':
        detail = '学生控制' if t < 6 else '专家已接管'
    elif scene_kind == 'expert':
        detail = '专家控制'
    else:
        detail = '学生自主控制'
    draw.text((20, 45), f'{detail}  ·  前进指令 20 cm/s  ·  黄色线标出阶沿', font=SMALL, fill='#c7d4e3')
    # Fixed legend: labels remain legible when feet occlude the physical edge.
    for i, color in enumerate(COLORS):
        left = 20+i*312
        draw.rounded_rectangle((left, HEADER+12, left+296, HEADER+48), 7, fill='#111a28', outline=color, width=2)
        draw.rectangle((left+10, HEADER+22, left+24, HEADER+36), fill=color)
        draw.text((left+34, HEADER+17), f'第{i+1}级  +{(i+1)*10} mm  /  x={EDGES[i]:.2f} m', font=SMALL, fill=color)
    draw.text((20, 648), f't = {t:5.2f} s     机身 x = {base_x:.3f} m     当前前进速度 {vx*100:+.1f} cm/s', font=FONT, fill='white')
    # A position diagram, not a reconstruction of individual foot placement.
    draw.text((20, 684), '位置示意', font=SMALL, fill='#c7d4e3')
    start, end = -.02, max(.96, base_x+.10)
    def sx(x): return int(134+(x-start)/(end-start)*800)
    bottom = 781
    bounds = (start,)+EDGES+(end,)
    for i in range(4):
        top = bottom-12-i*11
        draw.rectangle((sx(bounds[i]), top, sx(bounds[i+1]), bottom), fill='#536071' if i==0 else COLORS[i-1])
        if i:
            draw.line((sx(bounds[i]), top, sx(bounds[i]), bottom), fill='#ffe033', width=3)
            draw.text((sx(bounds[i])+5, 785), f'{bounds[i]:.2f} m', font=SMALL, fill='#c7d4e3')
    bx = sx(base_x)
    draw.line((bx, 722, bx, bottom), fill='white', width=2)
    draw.polygon(((bx-7, 717), (bx+7, 717), (bx, 728)), fill='white')
    text_x = max(134, min(bx-40, 828))
    draw.text((text_x, 689), '机身位置', font=SMALL, fill='white')
    return output
