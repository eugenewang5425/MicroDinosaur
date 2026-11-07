"""Restore the tested IMX219 media mode after a reboot, before capture starts."""
import re
import subprocess


SENSOR = 'm00_b_imx219 2-0010'


def read_format(text):
    match = re.search(r'fmt:([A-Z0-9_]+)/([0-9]+)x([0-9]+)', text)
    if match is None:
        raise RuntimeError('Unable to read the CSI media format: ' + text.strip())
    return match.group(1), int(match.group(2)), int(match.group(3))


def configure_sensor(mode='1920x1080', run=subprocess.run):
    """Set only runtime media formats; no boot, exposure or clock changes.

    Verify every pad even when the sensor already reports the desired size:
    partially configured links must not silently survive a service restart.
    """
    records = []

    def command(args):
        result = run(args, capture_output=True, text=True, timeout=12)
        records.append({'args': args, 'returncode': result.returncode,
                        'stdout': result.stdout, 'stderr': result.stderr})
        if result.returncode:
            raise RuntimeError('CSI mode command failed: ' + repr(records[-1]))
        return result.stdout

    pads = [(SENSOR, 0, 'SRGGB10_1X10'),
            ('rockchip-csi2-dphy0', 0, 'SRGGB10_1X10'),
            ('rkisp-csi-subdev', 0, 'SRGGB10_1X10'),
            ('rkisp-isp-subdev', 0, 'SRGGB10_1X10'),
            ('rkisp-isp-subdev', 2, 'YUYV8_2X8')]
    before = [read_format(command(['media-ctl', '-d', '/dev/media0', '--get-v4l2',
                                   f'"{entity}":{pad}']))
              for entity, pad, _ in pads]
    if mode == 'current':
        return {'requested': mode, 'actual': list(before[0]), 'changed': False,
                'commands': records}
    if mode != '1920x1080':
        raise ValueError('Only the tested 1920x1080 mode is supported')
    target = [(code, 1920, 1080) for _, _, code in pads]
    changed = before != target
    if changed:
        for entity, pad, code in pads:
            crop = ' crop:(0,0)/1920x1080' if entity == 'rkisp-isp-subdev' else ''
            command(['media-ctl', '-d', '/dev/media0', '-V',
                     f'"{entity}":{pad}[fmt:{code}/1920x1080{crop}]'])
        # The existing 3A service must re-read the negotiated sensor mode.
        command(['systemctl', 'restart', 'rkaiq_3A'])
    after = [read_format(command(['media-ctl', '-d', '/dev/media0', '--get-v4l2',
                                  f'"{entity}":{pad}']))
             for entity, pad, _ in pads]
    if after != target:
        raise RuntimeError('CSI mode readback mismatch: ' + repr(after))
    return {'requested': mode, 'actual': list(after[0]), 'changed': changed,
            'previous': [list(value) for value in before], 'commands': records}
