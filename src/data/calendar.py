import re
from pathlib import Path
from datetime import datetime

PAT = re.compile(r"Yahoo_1m_(\d{2})_(\d{2})_(\d{2})$")

def list_day_directories(data_root: Path) -> list[Path]:
    dirs = []
    for p in data_root.iterdir():
        if p.is_dir() and PAT.match(p.name):
            dirs.append(p)
    # tri chronologique
    def parse_date(folder: Path):
        dd, mm, yy = PAT.match(folder.name).groups()
        return datetime(int("20"+yy), int(mm), int(dd))
    return sorted(dirs, key=parse_date)
