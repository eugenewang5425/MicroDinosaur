"""Read the planned JY61P on ZERO 3W I2C4 without changing its settings."""
import argparse
import ctypes
import datetime
import json
import math
import os
from pathlib import Path
import statistics
import struct
import time


class Message(ctypes.Structure):
    _fields_ = [("addr", ctypes.c_uint16), ("flags", ctypes.c_uint16),
                ("length", ctypes.c_uint16), ("buffer", ctypes.c_void_p)]


class Transfer(ctypes.Structure):
    _fields_ = [("messages", ctypes.POINTER(Message)), ("count", ctypes.c_uint32)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=5)
    parser.add_argument("--hz", type=float, default=50)
    parser.add_argument("--label", default="connection_check")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not 0 < args.seconds <= 30 or not 0 < args.hz <= 100:
        parser.error("Use at most 30 seconds and 100 register reads per second")
    if args.out.exists():
        parser.error("Output already exists; retain earlier measurements")

    libc = ctypes.CDLL(None, use_errno=True)
    libc.ioctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_void_p]
    libc.ioctl.restype = ctypes.c_int
    result = {
        "checked_at": datetime.datetime.now().astimezone().isoformat(),
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "bus": "/dev/i2c-4", "address_7bit": "0x50", "label": args.label,
        "sensor_settings_written": False,
        "timestamp_source": "host monotonic register-read time, not sensor sample time",
        "records": [],
    }
    fd = None
    try:
        adapter = Path("/sys/class/i2c-adapter/i2c-4").resolve()
        result["adapter_path"] = str(adapter)
        if "fe5d0000.i2c" not in str(adapter):
            raise RuntimeError("I2C4 does not resolve to the expected RK3566 controller")
        fd = os.open("/dev/i2c-4", os.O_RDWR | os.O_CLOEXEC)

        def read(register, size):
            # The write message selects the read register; it contains no value
            # and does not alter calibration, address, mode or saved settings.
            pointer = (ctypes.c_ubyte * 1)(register)
            data = (ctypes.c_ubyte * size)()
            messages = (Message * 2)(
                Message(0x50, 0, 1, ctypes.addressof(pointer)),
                Message(0x50, 1, size, ctypes.addressof(data)),
            )
            transfer = Transfer(messages, 2)
            rc = libc.ioctl(fd, 0x0707, ctypes.byref(transfer))
            if rc < 0:
                error = ctypes.get_errno()
                raise OSError(error, os.strerror(error))
            if rc != 2:
                raise RuntimeError(f"Only {rc} of 2 I2C messages completed")
            return bytes(data)

        result["address_register_0x1a"] = int.from_bytes(read(0x1A, 2), "little")
        result["version_register_0x2e"] = int.from_bytes(read(0x2E, 2), "little")
        started = time.monotonic()
        due = started
        while time.monotonic() - started < args.seconds:
            entry = {"t_s": time.monotonic() - started}
            before = time.monotonic()
            try:
                raw = read(0x34, 24)
                values = struct.unpack("<12h", raw)
                entry.update({
                    "raw_hex": raw.hex(),
                    "raw_i16": list(values),
                    "acc_g": [v * 16 / 32768 for v in values[:3]],
                    "gyro_dps": [v * 2000 / 32768 for v in values[3:6]],
                    "angles_deg": [v * 180 / 32768 for v in values[9:12]],
                    "ok": True,
                })
                entry["acc_norm_g"] = math.sqrt(sum(v*v for v in entry["acc_g"]))
            except Exception as error:
                entry.update(ok=False, error=repr(error))
            entry["read_ms"] = (time.monotonic() - before) * 1000
            result["records"].append(entry)
            due += 1 / args.hz
            time.sleep(max(0, due - time.monotonic()))
        result["elapsed_s"] = time.monotonic() - started
    except Exception as error:
        result["error"] = repr(error)
    finally:
        if fd is not None:
            os.close(fd)

    successful = [r for r in result["records"] if r.get("ok")]
    summary = {
        "attempts": len(result["records"]), "successes": len(successful),
        "errors": len(result["records"]) - len(successful),
        "connection_passed": bool(successful and result.get("address_register_0x1a") == 0x50),
    }
    if successful:
        summary["unique_payloads"] = len({r["raw_hex"] for r in successful})
        summary["acc_norm_g_median"] = statistics.median(r["acc_norm_g"] for r in successful)
        summary["gyro_dps_median"] = [statistics.median(r["gyro_dps"][i] for r in successful) for i in range(3)]
        summary["gyro_dps_range"] = [[min(r["gyro_dps"][i] for r in successful), max(r["gyro_dps"][i] for r in successful)] for i in range(3)]
        summary["read_ms_median"] = statistics.median(r["read_ms"] for r in successful)
        summary["first_sample"] = successful[0]
        summary["last_sample"] = successful[-1]
    result["summary"] = summary
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, ensure_ascii=False, indent=2))
    return 0 if summary["connection_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
