import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))


@pytest.fixture(autouse=True)
def _reset_stop():
    from genki2anki import runctl
    runctl.STOP.clear()
    yield
    runctl.STOP.clear()
