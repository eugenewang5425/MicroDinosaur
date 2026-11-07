import math
import struct
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path
import importlib.util
import sys
import s288_ftdi as s
import configure_eeprom as ee

def frame(mid=2,sensor=200,volts=24,position=-1234):
    body=struct.pack('<BbBBhhiIHBB',mid,45,sensor,volts,-10,20,position,0,2048,0,0)
    return b'\xfc\xee'+body+struct.pack('<I',s.crc32_words(body))

class Tests(unittest.TestCase):
    def test_known_official_stop_vector(self):
        self.assertEqual(s.command(0).hex(),'feee80000000000000000000000000001d99fb6e')
    def test_signed_position_and_no_wrap(self):
        for q in [-.123,0,.123]:
            packet=s.command(8,mode=1,position=q,kp=1.2,kd=.2)
            self.assertEqual(struct.unpack('<i',packet[8:12])[0],round(q*s.RATIO*32768/(2*math.pi)))
        for q in [float('nan'),float('inf'),1e9]:
            with self.assertRaises(ValueError):s.command(0,mode=1,position=q)
    def test_unicast_timeout_and_gain_guards(self):
        for mid in [-1,15,True]:
            with self.assertRaises(ValueError):s.command(mid)
        with self.assertRaises(ValueError):s.command(0,mode=1,timeout=0)
        with self.assertRaises(ValueError):s.command(0,mode=1,kp=-1)
    def test_unsigned_feedback_and_signed_position(self):
        got=s.decode(frame())
        self.assertEqual((got.sensor,got.voltage_v,got.temperature_c),(200,12,45))
        self.assertLess(got.position_rad,0)
    def test_fragmentation_noise_and_crc_recovery(self):
        good=frame();bad=bytearray(good);bad[8]^=1
        reader=s.FrameReader();self.assertEqual(reader.feed(b'noise\xfc'),[])
        self.assertEqual(reader.feed(b'\x00'+bad+good[:11]),[])
        self.assertEqual([x.motor_id for x in reader.feed(good[11:]+frame(3))],[2,3])
        self.assertEqual(reader.pending,b'')
        self.assertEqual(reader.bad_crc,1)
    def test_ports_bind_by_usb_interface_not_tty_order(self):
        ports=[s.Port('/dev/ttyUSB9','ONE',1),s.Port('/dev/ttyUSB3','ONE',0),s.Port('/dev/ttyUSB0','TWO',0)]
        self.assertEqual(s.bind_ports(ports,'ONE'),{'A':'/dev/ttyUSB3','B':'/dev/ttyUSB9'})
        with self.assertRaises(ValueError):s.bind_ports(ports+[ports[0]],'ONE')
        with self.assertRaises(ValueError):s.bind_ports(ports,'UNKNOWN')
    def test_construction_opens_nothing_and_partial_response(self):
        created=[]
        class Fake:
            def __init__(self,**kw):self.kw=kw;self.buffer=bytearray();self.closed=False;created.append(self)
            def reset_input_buffer(self):self.buffer.clear()
            def write(self,data):self.buffer.extend(frame(data[2]&15));return len(data)
            @property
            def in_waiting(self):return len(self.buffer)
            def read(self,size):result=bytes(self.buffer[:3]);del self.buffer[:3];return result
            def close(self):self.closed=True
        bus=s.Bus('FAKE',serial_factory=Fake)
        self.assertEqual(created,[])
        bus.open();got=bus.request(2);self.assertEqual(got.motor_id,2)
        self.assertEqual(created[0].kw['baudrate'],6000000)
        self.assertFalse(created[0].kw['rtscts']);bus.close()
    def test_timeout_closes_transport(self):
        fake=SimpleNamespace(reset_input_buffer=lambda:None,write=lambda x:len(x),in_waiting=0,read=lambda n:b'',close=lambda:None)
        bus=s.Bus('FAKE',deadline_s=0,serial_factory=lambda **kw:fake);bus.open()
        with self.assertRaises(TimeoutError):bus.request(1)
        self.assertIsNone(bus.serial)
    def test_eeprom_plan_has_two_uart_and_host_power_gating(self):
        plan=ee.settings('MDHB20001');p=plan['properties']
        self.assertEqual(plan['eeprom_bytes'],256)
        self.assertTrue(p['self_powered'] and p['suspend_dbus7'])
        self.assertEqual((p['channel_a_type'],p['channel_b_type']),('UART','UART'))
        for serial in ['', 'bad serial', 'a'*17]:
            with self.assertRaises(ValueError):ee.settings(serial)
    def test_io_failure_closes_and_reopen_drops_old_frame(self):
        closed=[]
        def failed_write(data):raise OSError('device disconnected')
        fake=SimpleNamespace(reset_input_buffer=lambda:None,write=failed_write,close=lambda:closed.append(True))
        bus=s.Bus('FAKE',serial_factory=lambda **kw:fake)
        bus.reader.feed(frame()[:12]);bus.open()
        self.assertEqual(bus.reader.pending,b'')
        with self.assertRaises(OSError):bus.request(1)
        self.assertIsNone(bus.serial);self.assertEqual(closed,[True])

if __name__=='__main__':unittest.main(verbosity=2)
