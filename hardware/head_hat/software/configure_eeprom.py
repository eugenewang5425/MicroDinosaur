"""Generate the B2 configuration plan by default; explicit --apply writes hardware.
PyFtdi API reference: https://github.com/eblot/pyftdi/blob/master/pyftdi/eeprom.py
Writing is only for an assembled, electrically verified board without motors.
"""
import argparse,json
from pathlib import Path

def settings(serial_number):
    if not serial_number or len(serial_number)>16 or not serial_number.isascii() or not serial_number.isalnum():
        raise ValueError('Use a unique ASCII alphanumeric serial, at most 16 characters')
    return {'serial_number':serial_number,'manufacturer':'MicroDinosaur','product':'HEAD B2 Dual S288',
            'properties':{'vendor_id':0x0403,'product_id':0x6010,'chip':0x56,'self_powered':True,'power_max':0,
                          'remote_wakeup':False,'suspend_dbus7':True,'channel_a_type':'UART','channel_b_type':'UART',
                          'channel_a_driver':'VCP','channel_b_driver':'VCP'},'eeprom_model':'93LC56B','eeprom_bytes':256}

def configure(eeprom,plan):
    # Reset the in-memory image only; commit below is the sole hardware write.
    # Starting from blank FF otherwise preserves undefined drive/mode bits.
    eeprom.erase(0)
    # USB configuration descriptor bmAttributes bit 7 is required to be one.
    # Pinned PyFtdi 0.57.1 preserves it rather than initializing it. This is
    # an in-memory image edit; test_eeprom_image checks the serialized byte.
    eeprom._eeprom[0x08] = 0x80
    eeprom.initialize()
    eeprom.set_manufacturer_name(plan['manufacturer'])
    eeprom.set_product_name(plan['product'])
    eeprom.set_serial_number(plan['serial_number'])
    for key,value in plan['properties'].items():
        eeprom.set_property(key,value)
    eeprom.sync()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--serial',required=True)
    parser.add_argument('--url',help='Explicit pyftdi URL selecting this physical FT2232H')
    parser.add_argument('--output',type=Path,default=Path('eeprom_plan.json'))
    parser.add_argument('--apply',action='store_true')
    args=parser.parse_args();plan=settings(args.serial)
    args.output.write_text(json.dumps(plan,indent=2),encoding='utf-8')
    if not args.apply:
        print('Offline plan saved. No USB device was opened or written.');return
    if not args.url or not args.url.startswith('ftdi://'):
        parser.error('--apply requires the explicit physical device URL')
    from pyftdi.eeprom import FtdiEeprom
    e=FtdiEeprom()
    try:
        e.open(args.url,model='93LC56B')
        if e.device_version != 0x0700:
            raise RuntimeError('Selected chip is not FT2232H')
        args.output.with_suffix('.before.bin').write_bytes(e.data)
        configure(e,plan)
        args.output.with_suffix('.planned.bin').write_bytes(e.data)
        # Library commit reads EEPROM back and compares every byte after writing.
        e.commit(dry_run=False)
        args.output.with_suffix('.verified.bin').write_bytes(e.data)
        print('EEPROM written and byte-compared by PyFtdi. Power-cycle, enumerate both UARTs, and verify suspend gating before use.')
    finally:
        e.close()

if __name__=='__main__':main()
