"""Import-time bootstrap that makes ``ai4us`` importable from any experiment script.

Every experiment module starts with::

    from experiments import _bootstrap  # noqa: F401

This guarantees that running a script directly with ``python
experiments/01_.../generate.py`` works without first installing the
package. When the repo has been ``pip install``-ed the bootstrap is a
no-op.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
