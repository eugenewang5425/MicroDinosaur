"""HEAD B2 host transport. Importing this module never opens a device.

Protocol: Unitree digital_servo 75c1ed63579dc7bb759e2990bd4b804c11b57daf
and J288/S288 user manual. Positions/velocities/torques use output-shaft units.
Existing robot zero, direction and joint mapping belong to the caller.
"""
from dataclasses import dataclass
import math
from pathlib import Path
import struct
import threading
import time

RATIO = 70070.0 / 243.0
BAUD = 6_000_000
VID_PID = (0x0403, 0x6010)

def crc32_words(data):
    if len(data) % 4:
        raise ValueError('CRC input must contain whole little-endian words')
    crc = 0xffffffff
    for offset in range(0, len(data), 4):
        crc ^= int.from_bytes(data[offset:offset+4], 'little')
        for _ in range(32):
            crc = ((crc << 1) ^ (0x04c11db7 if crc & 0x80000000 else 0)) & 0xffffffff
    return crc

def _count(value, scale, low, high):
    if isinstance(value, bool) or not math.isfinite(value):
        raise ValueError('Finite numeric value required')
    count = round(value * scale)
    if not low <= count <= high:
        raise ValueError('Command exceeds protocol range')
    return count

def command(mid, *, mode=0, position=0.0, velocity=0.0, torque=0.0, kp=0.0, kd=0.0, timeout=1):
    if type(mid) is not int or not 0 <= mid <= 14:
        raise ValueError('Only unicast IDs 0..14 are supported')
    if mode not in (0, 1) or type(mode) is not int or timeout not in (0, 1) or type(timeout) is not int:
        raise ValueError('Unsupported control mode/timeout')
    if mode and not timeout:
        raise ValueError('Motion requires servo timeout protection')
    if kp < 0 or kd < 0:
        raise ValueError('Negative gains are not supported')
    if not mode:
        position = velocity = torque = kp = kd = 0.0
    fields = (_count(torque, 256000/RATIO, -32768, 32767),
              _count(velocity, RATIO*2.56/(2*math.pi), -32768, 32767),
              _count(position, RATIO*32768/(2*math.pi), -2147483648, 2147483647),
              _count(kp, 1280000/(RATIO*RATIO), 0, 32767),
              _count(kd, 128000000/(RATIO*RATIO), 0, 32767))
    data = struct.pack('<BBBBhhihh', 0xfe, 0xee, mid | mode << 4 | timeout << 7, 0, *fields)
    return data + struct.pack('<I', crc32_words(data))

@dataclass(frozen=True)
class Feedback:
    motor_id: int
    mode: int
    timed_out: bool
    temperature_c: int
    sensor: int
    voltage_v: float
    torque_nm: float
    velocity_rad_s: float
    position_rad: float
    errors: int
    warnings: int
    output_encoder_rad: float
    raw: bytes

def decode(frame):
    if len(frame) != 26 or frame[:2] != b'\xfc\xee':
        raise ValueError('Bad feedback frame')
    if crc32_words(frame[2:22]) != int.from_bytes(frame[22:], 'little'):
        raise ValueError('Bad feedback CRC')
    status, temp, sensor, volts, tor, vel, pos, errors, external, _, _ = struct.unpack('<BbBBhhiIHBB', frame[2:22])
    return Feedback(status & 15, status >> 4 & 7, bool(status & 128), temp, sensor, volts/2,
                    tor*RATIO/256000, vel*2*math.pi/(2.56*RATIO), pos*2*math.pi/(32768*RATIO),
                    errors, external >> 13, (external & 8191)*2*math.pi/8192, bytes(frame))

class FrameReader:
    def __init__(self):
        self.pending = bytearray()
        self.bad_crc = 0
        self.discarded = 0
    def feed(self, chunk):
        self.pending.extend(chunk)
        frames = []
        while len(self.pending) >= 2:
            index = self.pending.find(b'\xfc\xee')
            if index < 0:
                keep = 1 if self.pending[-1] == 0xfc else 0
                self.discarded += len(self.pending)-keep
                self.pending[:] = b'\xfc' if keep else b''
                break
            self.discarded += index
            del self.pending[:index]
            if len(self.pending) < 26:
                break
            try:
                frame = decode(bytes(self.pending[:26]))
            except ValueError:
                self.bad_crc += 1
                del self.pending[0]
            else:
                frames.append(frame)
                del self.pending[:26]
        return frames

@dataclass(frozen=True)
class Port:
    device: str
    serial_number: str
    interface_number: int
    vid: int = VID_PID[0]
    pid: int = VID_PID[1]

def bind_ports(ports, serial_number):
    if not serial_number:
        raise ValueError('Explicit board serial number required')
    selected = {}
    for interface, bus in [(0, 'A'), (1, 'B')]:
        matches = [p for p in ports if (p.vid,p.pid) == VID_PID and p.serial_number == serial_number and p.interface_number == interface]
        if len(matches) != 1:
            raise ValueError(f'Bus {bus} requires exactly one matching USB interface; found {len(matches)}')
        selected[bus] = matches[0].device
    if selected['A'] == selected['B']:
        raise ValueError('Both buses mapped to the same port')
    return selected

def inventory_linux(sys_root=Path('/sys/class/tty')):
    """Read USB identity at the physical sysfs parents, not tty numbering."""
    ports=[]
    for tty in Path(sys_root).glob('ttyUSB*'):
        device=(tty/'device').resolve()
        chain=[device,*device.parents]
        usb=next((x for x in chain if (x/'idVendor').is_file() and (x/'idProduct').is_file()),None)
        interface=next((x for x in chain if (x/'bInterfaceNumber').is_file()),None)
        if usb is None or interface is None or not (usb/'serial').is_file():
            continue
        ports.append(Port('/dev/'+tty.name,(usb/'serial').read_text().strip(),int((interface/'bInterfaceNumber').read_text().strip(),16),int((usb/'idVendor').read_text().strip(),16),int((usb/'idProduct').read_text().strip(),16)))
    return ports

class Bus:
    def __init__(self, port, *, deadline_s=.02, serial_factory=None):
        self.port=port
        self.deadline_s=deadline_s
        self.factory=serial_factory
        self.serial=None
        self.reader=FrameReader()
        self.lock=threading.Lock()
    def open(self):
        if self.serial is not None:
            raise RuntimeError('Already open')
        factory=self.factory
        if factory is None:
            from serial import Serial
            factory=Serial
        self.serial=factory(port=self.port,baudrate=BAUD,bytesize=8,parity='N',stopbits=1,timeout=.001,write_timeout=.02,xonxoff=False,rtscts=False,dsrdtr=False)
        try:
            self.serial.reset_input_buffer()
            self.reader=FrameReader()
        except Exception:
            self.close()
            raise
    def close(self):
        if self.serial is not None:
            device,self.serial=self.serial,None
            device.close()
    def request(self, mid, **target):
        packet=command(mid,**target)
        with self.lock:
            if self.serial is None:
                raise RuntimeError('Port has not been opened')
            # One outstanding request per bus; a timed-out bus is closed below.
            try:
                if self.serial.write(packet) != len(packet):
                    raise IOError('Partial serial write')
                deadline=time.monotonic()+self.deadline_s
                while time.monotonic()<deadline:
                    raw=self.serial.read(max(1,min(256,self.serial.in_waiting)))
                    for frame in self.reader.feed(raw):
                        if frame.motor_id != mid:
                            continue
                        if frame.errors:
                            raise IOError(f'Servo {mid} error 0x{frame.errors:x}')
                        return frame
                raise TimeoutError(f'No valid response for ID {mid}; reconnect explicitly')
            except Exception:
                self.close()
                raise

# These are testable protocol/transport building blocks, not a robot scheduler.
# Keep the existing project watchdog, profile, zero and direction mapping above
# this layer. Hardware TXDEN handles turnaround: do not toggle RTS in software.
