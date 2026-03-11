"""
Registry of golden eval test cases.
Single import point for all eval cases.
"""
import json
from pathlib import Path

_CASES_DIR = Path(__file__).parent

_CASE_FILES = [
    "case_boeing.json",
    "case_nasa.json",
    "case_mayo_clinic.json",
    "case_remora_carbon.json",
    "case_ather_energy.json",
]

# Load all cases at module import time
GOLDEN_CASES: dict[str, dict] = {}
for _filename in _CASE_FILES:
    _path = _CASES_DIR / _filename
    with open(_path, "r", encoding="utf-8") as _f:
        _case = json.load(_f)
    GOLDEN_CASES[_case["id"]] = _case


def get_case(case_id: str) -> dict:
    """
    Return the golden test case for the given ID (e.g. "TC-01").

    Raises:
        KeyError: If case_id is not registered.
    """
    if case_id not in GOLDEN_CASES:
        available = ", ".join(sorted(GOLDEN_CASES.keys()))
        raise KeyError(f"Unknown case ID '{case_id}'. Available: {available}")
    return GOLDEN_CASES[case_id]
