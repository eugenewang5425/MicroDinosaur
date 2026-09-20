"""Run the snapshot's existing 14 head-controller checks against its frozen plant."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'training/src'))
sys.path.insert(0, str(ROOT / 'research'))

if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromNames(['test_head_attitude', 'test_imu_heading'])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
