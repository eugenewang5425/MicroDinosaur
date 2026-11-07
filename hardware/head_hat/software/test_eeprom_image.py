"""Exercise real pinned PyFtdi serializer without opening USB."""
from types import SimpleNamespace
from pathlib import Path
import unittest
from pyftdi.eeprom import FtdiEeprom
from pyftdi.ftdi import Ftdi
import configure_eeprom as cfg

class ImageTest(unittest.TestCase):
    def test_full_image_and_descriptor_decode(self):
        fake=SimpleNamespace(device_version=0x0700,is_connected=True,device_name='ft2232h')
        fake.calc_eeprom_checksum=lambda data:Ftdi.calc_eeprom_checksum(fake,data)
        ee=FtdiEeprom();ee._ftdi=fake;ee._dev_ver=0x0700;ee._size=256
        ee.erase(255);cfg.configure(ee,cfg.settings('MDHB20001'))
        image=ee.data
        self.assertEqual(len(image),256)
        self.assertEqual(image[8],0xc0)  # mandatory bit 7 + self-powered bit 6
        self.assertEqual(int.from_bytes(image[-2:],'little'),fake.calc_eeprom_checksum(image[:-2]))
        self.assertEqual(ee._config['chip'],0x56)
        self.assertTrue(ee._config['self_powered'] and ee._config['suspend_dbus7'])
        self.assertEqual(ee._config['serial'],'MDHB20001')
        for ch in 'ab':
            self.assertEqual(ee._config[f'channel_{ch}_type'],'UART')
            self.assertEqual(ee._config[f'channel_{ch}_driver'],'VCP')
        self.assertFalse(ee._config['suspend_pull_down'])

if __name__=='__main__':unittest.main(verbosity=2)
