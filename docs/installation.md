# Installation

## Requirements

Python >= 3.11. Core dependencies (installed automatically): `numpy`, `pandas`,
`numba`, `scipy`, `finufft`.

## Recommended: install from a local clone

```bash
git clone https://github.com/jecampagne/nufftcf.git
cd nufftcf
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install --upgrade pip

# Force prebuilt wheels for the three compiled dependencies.
# This avoids source builds that can fail ("Failed to build installable
# wheels for some pyproject.toml based projects: finufft, llvmlite") or
# produce mismatched OpenMP runtimes, particularly on macOS
# (see Troubleshooting below).
pip install --only-binary=:all: finufft numba llvmlite

# installs nufftcf itself, plus dev, test and benchmark extras in one go
pip install -e ".[dev,test,benchmark]"
```

!!! tip
    Always run the `--only-binary=:all:` step in a **fresh** virtual
    environment, before installing anything else. If you already tried a
    plain `pip install -e .` that failed to build `finufft` or `llvmlite`,
    delete the venv and start over (`rm -rf venv`) rather than re-running
    the wheel-only install on top of a partially-failed environment.

## Install directly from GitHub

```bash
# recommended: force prebuilt wheels first, in a fresh venv (see note below)
pip install --only-binary=:all: finufft numba llvmlite

pip install "nufftcf @ git+https://github.com/jecampagne/nufftcf.git"
# with benchmark extras:
pip install "nufftcf[benchmark] @ git+https://github.com/jecampagne/nufftcf.git"
```

!!! note
    For maximum reliability, always run
    `pip install --only-binary=:all: finufft numba llvmlite` first, in a
    fresh virtual environment -- particularly on macOS. See
    [Troubleshooting](#troubleshooting-macos) if you hit a build failure
    for `finufft` or `llvmlite`.

## Troubleshooting (macOS)

Two distinct failure modes are common on macOS, and both have the same fix.

**"Failed to build installable wheels for some pyproject.toml based
projects: finufft, llvmlite"** at `pip install` time -- pip is trying to
compile `finufft` (C++) and/or `llvmlite` (LLVM bindings) from source
instead of using a prebuilt wheel, and the source build fails (missing
compiler toolchain, incompatible LLVM version, etc.).

**A segfault or hang** when running tests or importing the package --
usually a mismatched OpenMP runtime between a source-built `finufft` and
the `numba`/`llvmlite` stack, or a stale Numba on-disk JIT cache from a
previous, differently-built install.

The short version: make sure you have a clean Python environment (no
conda stacked on top of your venv), install `finufft`/`numba`/`llvmlite`
as prebuilt wheels (`--only-binary=:all:`) **in a fresh venv**, and clear
any stale Numba on-disk JIT cache:

```bash
deactivate 2>/dev/null   # if a venv is currently active
rm -rf venv ~/.numba_cache
find . -type d -name "__pycache__" -exec rm -rf {} +

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install --only-binary=:all: finufft numba llvmlite
pip install -e ".[dev,test,benchmark]"
pytest tests/
```
