"""Training CLI compatibility entry point for ``python -m train``."""

from training import run


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)
