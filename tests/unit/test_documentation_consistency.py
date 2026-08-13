import importlib.util
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location("documentation_consistency", _ROOT / "scripts/check_documentation_consistency.py")
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
check = _MODULE.check


def test_current_documentation_metadata_is_consistent() -> None:
    assert check(_ROOT) == []
