"""Query the F411 bench firmware. Reading only; never changes IMU registers."""
from pathlib import Path
import argparse, sys, json, datetime, time, re, struct, math, statistics

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'deps'))
import serial
from serial.tools import list_ports

REASONS = {0:'OK',1:'BUS_LOW_BEFORE_START',2:'SCL_TIMEOUT_AT_START',
           3:'ADDRESS_WRITE_NACK',4:'REGISTER_NACK',5:'ADDRESS_READ_NACK',
           6:'READ_CLOCK_TIMEOUT',13:'SCL_TIMEOUT_DURING_WRITE_ADDRESS',
           14:'SCL_TIMEOUT_DURING_REGISTER',15:'SCL_TIMEOUT_DURING_READ_ADDRESS'}

def query(s, command):
    s.write((command+'\n').encode('ascii'))
    line = s.readline().decode('ascii',errors='replace').strip()
    item={'host_time':datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'command':command,'line':line}
    for k,v in re.findall(r'(\w+)=([^ ]+)',line): item[k]=v
    if 'rc' in item:
        item['reason']=REASONS.get(int(item['rc']),'UNKNOWN')
    if item.get('reg')=='34' and item.get('rc')=='0':
        raw=bytes.fromhex(item['raw'])
        if len(raw)!=24: raise ValueError('Wrong data length')
        vals=struct.unpack('<12h',raw)
        item['registers']=list(vals)
        item['acc_g']=[v*16/32768 for v in vals[:3]]
        item['gyro_dps']=[v*2000/32768 for v in vals[3:6]]
        item['angles_deg']=[v*180/32768 for v in vals[9:12]]
        item['gravity_norm_g']=math.sqrt(sum(v*v for v in item['acc_g']))
    return item

def summarize(records):
    data=[r for r in records if r['command']=='D']
    valid=[r for r in data if r.get('rc')=='0']
    result={'attempts':len(data),'valid_reads':len(valid),
            'errors':{k:sum(r.get('reason','NO_REPLY')==k for r in data) for k in sorted({r.get('reason','NO_REPLY') for r in data})}}
    if valid:
        result['gravity_norm_g_mean']=statistics.mean(r['gravity_norm_g'] for r in valid)
        for key in ['acc_g','gyro_dps','angles_deg']:
            result[key]={'mean':[statistics.mean(r[key][i] for r in valid) for i in range(3)],
                         'min':[min(r[key][i] for r in valid) for i in range(3)],
                         'max':[max(r[key][i] for r in valid) for i in range(3)]}
        result['unique_payloads']=len({r['raw'] for r in valid})
        result['all_zero_payloads']=sum(not any(r['registers']) for r in valid)
        result['note']='Host register polls. Poll rate does not establish the IMU internal update rate.'
    return result

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('mode',choices=['status','diag','capture'])
    ap.add_argument('--port')
    ap.add_argument('--seconds',type=float,default=10)
    ap.add_argument('--weak-pullup',action='store_true')
    ap.add_argument('--fast',action='store_true')
    ap.add_argument('--half-us',type=int,choices=range(1,10))
    ap.add_argument('--quiet',action='store_true')
    ap.add_argument('--out',default=None)
    a=ap.parse_args()
    ports=[p for p in list_ports.comports() if p.vid==0x0483 and p.pid==0x5740 and p.serial_number=='358837723534']
    if not a.port and len(ports)!=1:raise SystemExit(f'Expected one F411 CDC port; found {ports}')
    port=a.port or ports[0].device
    records=[]
    pull_mode='none'
    def run(s,cmd):
        nonlocal pull_mode
        r=query(s,cmd)
        if 'pull' in r:pull_mode=r['pull']
        r['pull_mode']=pull_mode
        records.append(r)
        if not a.quiet or cmd!='D':print(r['line'],flush=True)
        return r
    with serial.Serial(port,115200,timeout=1) as s:
        s.reset_input_buffer()
        try:
            state=run(s,'S')
            if a.mode!='status' and state.get('v') not in ['2','3']:raise SystemExit('Diagnostic firmware v2/v3 required')
            if a.fast or a.half_us:
                if state.get('v')!='3':raise SystemExit('Firmware v3 required for --fast')
                run(s,str(a.half_us) if a.half_us else 'T')
            if a.mode=='diag':
                for cmd in ['N','S','L','D','D','P','S','L','I','V','D','D','D','D','D']:run(s,cmd);time.sleep(.1)
            elif a.mode=='capture':
                run(s,'P' if a.weak_pullup else 'N')
                run(s,'I');run(s,'V')
                until=time.monotonic()+a.seconds
                while time.monotonic()<until:
                    run(s,'D');time.sleep(.04)
        finally:
            if a.mode!='status':
                run(s,'N')
                if state.get('v')=='3':run(s,'H')
    report={'port':port,'mode':a.mode,'weak_pullup_requested':a.weak_pullup,'fast_requested':a.fast,'half_us_requested':a.half_us,'records':records,'summary':summarize(records),
            'by_pull_mode':{mode:summarize([r for r in records if r['pull_mode']==mode]) for mode in sorted({r['pull_mode'] for r in records})}}
    name=a.out or (a.mode+'_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'.json')
    dest=Path(name) if Path(name).is_absolute() else ROOT/name
    dest.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(report['summary'],indent=2));print('Saved',dest)

if __name__=='__main__':main()
