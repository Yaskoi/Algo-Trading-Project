from __future__ import annotations
from pathlib import Path
import re


DAY_RE = re.compile(r"Yahoo_1m_\d{2}_\d{2}_\d{2}$")


def list_day_directories(data_root: Path):
    if not data_root.exists():
        return []
    days = []
    for p in sorted(data_root.iterdir()):
        if p.is_dir() and DAY_RE.search(p.name):
            days.append(p)
    return days
