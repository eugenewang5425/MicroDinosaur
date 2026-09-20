"""Output-side numerical resolutions from Unitree's J288/S288 protocol.

Source: https://github.com/unitreerobotics/digital_servo/blob/main/specs/protocol.md
These are data-format resolutions, not mechanical accuracy or torque ratings.
"""
import math

GEAR_RATIO = 70070.0 / 243.0
OUTPUT_ENCODER_STEP = 2 * math.pi / 8192.0
ROTOR_POSITION_OUTPUT_STEP = 2 * math.pi / (32768.0 * GEAR_RATIO)
VELOCITY_STEP = 2 * math.pi / (2.56 * GEAR_RATIO)
TORQUE_STEP = GEAR_RATIO / 256000.0
KP_STEP = GEAR_RATIO**2 / 1280000.0
KD_STEP = GEAR_RATIO**2 / 128000000.0
JY61P_GYRO_STEP = math.radians(.061)
JY61P_GYRO_LIMIT = math.radians(2000)


def round_scalar(value, step):
    return round(value / step) * step


def wire_time_seconds(servo_count):
    """Wire-only request/response floor; excludes turnaround/firmware/scheduling."""
    if servo_count < 0:
        raise ValueError('servo_count must be nonnegative')
    return servo_count * (20 + 26) * 10 / 6000000.0
