from training import run

if __name__ == "__main__":
    success = run()
    raise SystemExit(0 if success else 1)
