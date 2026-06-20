# Packaging TACET as a standalone app

Goal: ship a self-contained app that runs without the user installing Python.

## 1. Develop inside a venv (always)

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python frontend/ui.py
```

Keep `(.venv)` active for every command. In VS Code, select this interpreter so
the integrated terminal activates it automatically.

## 2. Build the standalone app (when ready)

```
.venv\Scripts\activate
pip install pyinstaller
pyinstaller --noconfirm --windowed --name TACET ^
  --add-data "frontend/web;frontend/web" ^
  frontend/ui.py
```

Output lands in `dist/TACET/`. The `--add-data` flag bundles the HTML/CSS/JS UI;
we'll add more `--add-data` lines for the model files (below).

## 3. The big caveat: model weights

A truly offline standalone must ship the model weights, not download them on first
run. Plan for this now:

- **BERT**: download `bert-base-uncased` once, save it to a local folder
  (`models/bert/`), and load it from that path. Bundle the folder with
  `--add-data "models/bert;models/bert"`.
- **Whisper (faster-whisper)**: same idea — pre-download the chosen model size to
  `models/whisper/` and point the code at the local path, then bundle it.
- **TACET head**: ship `artifacts/tacet_head.pt` with `--add-data`.

This is why the bundle is large (~1–2 GB with torch + both models). That's the
trade-off for "no internet, no Python, just run it."

## 4. Size tips

- Use the **CPU-only** torch build (see requirements.txt) — much smaller than CUDA.
- Use a **small** Whisper model (`base` or `small`) unless accuracy demands more.
- Consider UPX compression with PyInstaller if size is a problem.
