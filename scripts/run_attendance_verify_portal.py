#!/usr/bin/env python3
"""Launch the attendance enrollment verification Flask portal."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")
load_dotenv(ROOT / ".env.publish", override=False)

from attendance_verify.app import main

if __name__ == "__main__":
    main()
