import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'src'))


def pytest_configure(config):
    # keep every default location inside a temporary work directory
    work = Path(config.rootpath) / '.pytest_cache' / 'work'
    os.environ.setdefault('NEXTACT_WORK', str(work))
