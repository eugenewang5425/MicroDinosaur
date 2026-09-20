"""姿态栈指令台(交互窗口):拖滑块实时检查全部可用指令。

启动即开窗(校准在后台线程,完成后滑块生效)——此前校准阻塞在窗口之前,
看起来像"没渲染"。通道即 pose_stack.CHANNELS:
  深蹲幅度 body-z 0~-25mm · 折叠蹲比例 0~100% · 尾俯仰/尾偏航
  · 左臂/右臂 · 脖子俯仰 · 嘴
头部俯仰/偏航由 IMU 稳定环托管(yaw_follows_trunk),不在命令栈内;
步态(跑模块 vx/wz)独立模块,另行调遣。
"""
from pathlib import Path
import sys
import threading
import tkinter as tk
from tkinter import ttk

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

D2R = 180. / 3.141592653589793
SLIDERS = [
    ('body_z',     -25.,   0.,  '{:+.1f} mm', lambda v: v / 1000.),
    ('fold',         0., 100.,  '{:.0f} %',   lambda v: v / 100.),
    ('tail_pitch', -26.,  26.,  '{:+.0f}°',   lambda v: v / D2R),
    ('tail_yaw',   -32.,  32.,  '{:+.0f}°',   lambda v: v / D2R),
    ('arm_l',      -37.,  37.,  '{:+.0f}°',   lambda v: v / D2R),
    ('arm_r',      -37.,  37.,  '{:+.0f}°',   lambda v: v / D2R),
    ('neck_pitch', -10.,  16.,  '{:+.0f}°',   lambda v: v / D2R),
    ('jaw',          0.,  14.,  '{:.1f}° 开嘴', lambda v: v / D2R),
]
NAMES = {'body_z': '深蹲幅度', 'fold': '折叠蹲比例', 'tail_pitch': '尾俯仰',
         'tail_yaw': '尾偏航', 'arm_l': '左臂', 'arm_r': '右臂',
         'neck_pitch': '脖子俯仰', 'jaw': '嘴'}


def main():
    root = tk.Tk()
    root.title('MicroDinosaur 姿态栈指令台(零训练 · 全部为指令)')
    root.geometry('560x780')

    status = ttk.Label(root, text='校准中(约 10 秒)…滑块暂时无效',
                       font=('msyh', 11, 'bold'), foreground='#b8860b')
    status.pack(fill=tk.X, padx=14, pady=(8, 2))

    vars_ = {}
    value_labels = {}
    slider_widgets = []
    for key, lo, hi, fmt, conv in SLIDERS:
        fr = ttk.Frame(root)
        fr.pack(fill=tk.X, padx=14, pady=3)
        ttk.Label(fr, text=NAMES[key], width=9, font=('msyh', 10)).pack(side=tk.LEFT)
        var = tk.DoubleVar(value=0. if key != 'jaw' else 2.3)
        vars_[key] = (var, fmt, conv)
        sl = ttk.Scale(fr, from_=lo, to=hi, variable=var, orient=tk.HORIZONTAL)
        sl.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        sl.config(state=tk.DISABLED)
        slider_widgets.append(sl)
        vl = ttk.Label(fr, text='', width=11, font=('msyh', 10))
        vl.pack(side=tk.RIGHT)
        value_labels[key] = vl

    btns = ttk.Frame(root)
    btns.pack(fill=tk.X, padx=14, pady=6)

    info = tk.Text(root, height=9, font=('msyh', 10), bg='#14171a', fg='#c8d2dc',
                   relief=tk.FLAT)
    info.pack(fill=tk.X, padx=14, pady=8)
    info.insert(tk.END, '校准中…')
    note = ttk.Label(root, text='步态 = 跑模块(vx 0.3-0.75 m/s / wz 转向率)独立模块。\n'
                                '起身 = 摔倒自动触发,不用指令;全折叠即坐撑,回 0% 自动站起。',
                     font=('msyh', 9), foreground='#8899aa')
    note.pack(fill=tk.X, padx=14)

    holder = {}

    def push_zero():
        for key, (var, fmt, conv) in vars_.items():
            var.set(2.3 if key == 'jaw' else 0.)

    def push_preset(which):
        zero = {k: (2.3 if k == 'jaw' else 0.) for k, (var, _, _) in vars_.items()}
        if which == 'squat':
            zero['body_z'] = -25.
        elif which == 'fold':
            zero['fold'] = 100.
        elif which == 'half':
            zero['fold'] = 50.
            zero['tail_pitch'] = -22.
            zero['arm_l'] = zero['arm_r'] = 12.
        for k, v in zero.items():
            vars_[k][0].set(v)

    def on_ready():
        status.config(text='就绪 — 拖动滑块即实时指令', foreground='#1a7a1a')
        for w in slider_widgets:
            w.config(state=tk.NORMAL)
        push_zero()
        tick()

    def calibrate():
        from pose_stack import PoseStack
        stack = PoseStack(941)
        holder['stack'] = stack
        root.after(0, on_ready)

    threading.Thread(target=calibrate, daemon=True).start()

    def tick():
        stack = holder.get('stack')
        if stack is None:
            root.after(100, tick)
            return
        cmd = {}
        for key, (var, fmt, conv) in vars_.items():
            v = var.get()
            cmd[key] = conv(v)
            value_labels[key].config(text=fmt.format(v))
        stack.apply(cmd)
        stack.step()
        stack.step()
        ro = stack.readout()
        active = '  '.join(f'{k}={stack.state[k]:+.3f}' for k in
                           ('body_z', 'fold', 'tail_pitch', 'tail_yaw',
                            'arm_l', 'arm_r', 'neck_pitch')
                           if abs(stack.state[k]) > 1e-6) or '全零'
        info.delete('1.0', tk.END)
        info.insert(tk.END,
                    f'sim t = {stack.s.data.time:7.2f} s\n'
                    f'根高   = {ro["root_z_mm"]:6.1f} mm\n'
                    f'倾角   = {ro["tilt_deg"]:5.2f}°\n'
                    f'双脚   = {ro["foot_n"]:5.2f} N   非足 = {ro["nonfoot_n"]:.2f} N\n'
                    f'通道   = {active}')
        root.after(40, tick)

    def wire_buttons():
        ttk.Button(btns, text='全部归零', command=push_zero).pack(side=tk.LEFT, padx=3)
        ttk.Button(btns, text='深蹲最大',
                   command=lambda: push_preset('squat')).pack(side=tk.LEFT, padx=3)
        ttk.Button(btns, text='折叠蹲 50%',
                   command=lambda: push_preset('half')).pack(side=tk.LEFT, padx=3)
        ttk.Button(btns, text='坐进折叠蹲',
                   command=lambda: push_preset('fold')).pack(side=tk.LEFT, padx=3)

    wire_buttons()
    root.mainloop()


if __name__ == '__main__':
    main()
