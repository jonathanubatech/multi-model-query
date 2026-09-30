#!/usr/bin/env python3
"""Multi-Model Query CLI — thin shim that delegates to ``multi_model_lib.cli``.

The implementation lives in the installable package (``multi_model_lib.cli``)
so the ``mmq`` console-script entry point works after ``pip install``. This
script is kept for direct invocation (``python scripts/multi-model-query.py``)
and simply forwards to the same ``main()``.
"""

from __future__ import annotations

import sys

from multi_model_lib.cli import main

if __name__ == "__main__":
    sys.exit(main())
