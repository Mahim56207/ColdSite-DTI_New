"""Make `from src...` imports work regardless of where pytest is invoked from.

Without this, `pytest tests/` works from the repo root and fails from inside
tests/, which is exactly the kind of thing that gets a suite quietly disabled.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
