import sys
from pathlib import Path

_root = Path(__file__).resolve().parent
_scripts = _root / "scripts"
_core = _scripts / "core"

if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))
if str(_core) not in sys.path:
    sys.path.insert(0, str(_core))
