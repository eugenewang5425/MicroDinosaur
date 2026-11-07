"""Inspect only by default. --apply sets FTDI latency timers for this board."""
import argparse,json
from pathlib import Path
from s288_ftdi import inventory_linux,bind_ports

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--serial',required=True)
    parser.add_argument('--apply',action='store_true')
    args=parser.parse_args();ports=bind_ports(inventory_linux(),args.serial)
    rows=[]
    for bus,device in ports.items():
        node=(Path('/sys/class/tty')/Path(device).name/'device').resolve()
        path=next((q/'latency_timer' for q in [node,*node.parents] if (q/'latency_timer').is_file()),None)
        if path is None:raise RuntimeError(f'No ftdi_sio latency_timer for {device}')
        before=int(path.read_text())
        if args.apply:
            path.write_text('1\n')
            if int(path.read_text())!=1:raise RuntimeError('Latency setting did not read back as 1 ms')
        rows.append({'bus':bus,'device':device,'serial':args.serial,'timer':str(path),'before_ms':before,'after_ms':int(path.read_text())})
    print(json.dumps({'changed':args.apply,'ports':rows,'serial_ports_opened':False},indent=2))

if __name__=='__main__':main()
