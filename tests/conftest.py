"""Make the package and the helper modules importable however pytest is invoked.

``python -m pytest`` puts the working directory on ``sys.path`` but a bare ``pytest``
does not, so the repository root is added explicitly. That keeps the documented
command working without installing the package, which matters because installing it
would pull in the full fitting stack that these tests deliberately avoid.
"""

import sys
from pathlib import Path

TESTS_DIR = Path(__file__).parent
REPO_ROOT = TESTS_DIR.parent

for path in (REPO_ROOT, TESTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
