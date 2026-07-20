"""
Force a safe Numba threading layer *before* numba (or anything importing
it) is loaded, for the test session only.

Why: `kernels.py` uses `@njit(parallel=True, ...)`, and `nufft_acf.py`
additionally calls FINUFFT (its own OpenMP thread pool) in the same call.
On some platforms -- macOS with Numba's TBB threading layer active is the
most commonly reported case -- running a `parallel=True` Numba function
and a separately-threaded library (FINUFFT) in the same process can
segfault or hang (see e.g. numba/numba#5973, numba/numba#7872). This has
nothing to do with the ACF math; it's strictly a threading-runtime
conflict. `workqueue` is Numba's simplest, dependency-free threading layer
and isn't affected by this class of issue.

This only affects `pytest` runs. It does NOT change anything about how
the package behaves for normal use (scripts, notebooks, your own code) --
if you hit the same symptoms outside of pytest, export the same
environment variables in your shell (or at the very top of your script,
before `import nufftcf`) -- see the README's "macOS troubleshooting"
note.

Override by setting these yourself in your environment before running
pytest -- this file only sets a default, via `setdefault`, it never
overwrites a value you've already exported.
"""

import os

os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
