"""Portable, CPU-only entrypoint; does not connect to robot hardware."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'training/src'))
sys.path.insert(0, str(ROOT / 'research'))

if __name__ == '__main__':
    from demo_head_attitude import main
    main()
