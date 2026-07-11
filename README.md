# TACET

**TACET** is a real-time conversational turn-taking system for voice assistants. It combines live speech recognition, semantic end-of-turn prediction, silence detection, and pitch analysis to decide when a user has finished speaking—helping an assistant respond naturally without interrupting.

## How it works

1. **Listen** — Capture microphone audio and transcribe it with Faster Whisper.
2. **Understand** — Encode the transcript with a frozen BERT encoder.
3. **Detect** — Estimate whether the utterance is complete with a trained neural classification head.
4. **Gate** — Combine the model score with silence, voice activity, and falling-pitch signals.
5. **Respond** — Generate a concise Gemini reply and speak it using Edge TTS.

## Features

- Live microphone transcription in a native desktop window
- Semantic end-of-turn detection instead of silence-only triggering
- Pitch-aware and spectrogram-based voice gating
- Local BERT and Whisper model support
- Configurable decision and silence thresholds
- Baseline and fine-tuning pipelines with plots and evaluation artifacts
- Transcript leak logging for later analysis
- Graceful fallbacks when Gemini or text-to-speech is unavailable

## Project structure

```text
TACET/
├── frontend/       # Desktop UI, browser assets, and live ASR
├── tacet/          # Encoder, inference, gating, LLM, and TTS logic
├── voice_mngt/     # Pitch and spectrogram processing
├── training/       # Training, fine-tuning, and visualization pipeline
├── testing/        # Unit, pipeline, and simulation tests
├── models/         # Optional local Whisper and BERT models
├── data/           # Training, validation, and test datasets
├── artifacts/      # Trained heads, metrics, and generated plots
├── config.py       # Shared paths, thresholds, and hyperparameters
└── requirements.txt
```

## Getting started

### Prerequisites

- Python 3.10 or newer
- A working microphone
- Internet access for Gemini and Edge TTS
- A Gemini API key for generated replies

### Installation

Create and activate a virtual environment, then install the dependencies (Inside the root folder).

**Windows (cmd):**

```bat
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Linux / macOS (bash):**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Create a `.env` file in the project root if you want Gemini responses:

```dotenv
GOOGLE_API_KEY=your_api_key_here
```

HOW TO GET THE API KEY: 
1. go to https://aistudio.google.com/apps
2. bottom left: click "Get API Key"
3. top right: click "Create API key"

Do not commit the `.env` file or any real API keys.

## Running the application

Launch the desktop voice interface from the project root (while in the venv):

```powershell
python frontend/ui.py
```

note: on your first run bert and whisper are going to download so it might take a while


## Training (ONLY IF YOU HAVE THE DATASET, WHICH YOU DO NOT BY DEFAULT. HOWEVER, THE NN CHECKPOINT IS IN THIS REPO SO YOU CAN RUN THE APP)

Run the baseline training and visualization pipeline:

```powershell
python -m training
```

Run the fine-tuning pipeline:

```powershell
python -m training finetune
```

The trained endpointer head is stored at `artifacts/finetune_tacet_head.pt`. Generated plots are written beneath `artifacts/plots/` or `artifacts/plots_finetune/`.

## Configuration

The main runtime and training settings live in `config.py`. Important options include:

| Setting | Purpose |
| --- | --- |
| `TAU` | Probability threshold for treating an utterance as complete |
| `MIN_THRESHOLD` | Standard silence duration before taking the turn |
| `PITCH_REQUIRED_SILENCE` | Shorter silence allowed after a falling-pitch signal |
| `SPEC_THRESHOLD` | Spectrogram/RMS threshold for detecting voice activity |
| `WHISPER_MODEL_SIZE` | Faster Whisper model used when no local model is present |
| `HEAD_PATH` | Path to the trained TACET classification head |

## Testing

Tests and end-to-end diagnostic scripts are located in `testing/`. Run an individual check from the project root, for example:
# EXAMPLE
python testing/infer_test.py
python testing/voice_pipeline.py


Some tests require model artifacts, audio hardware, or prepared datasets.

## Notes

- Gemini failures fall back to a short spoken availability message.
- Edge TTS requires a network connection; failed playback is printed to the terminal.
- Runtime behavior depends on the trained head and the thresholds in `config.py`.
- See `PACKAGING.md` for application packaging guidance.

---

# TASK SPLITTING SYNOPSIS:
Giorgos A. : 
tacet\__init__.py
tacet\data.py
tacet\encoder.py
tacet\infer.py
tacet\llm.py
tacet\model.py
tacet\preprocess.py
tacet\tts.py
frontend\__init__.py
frontend\asr.py
frontend\ui.py
frontend\web\app.js


Anastasios A. :
training\__init__.py
training\__main__.py
training\train.py
training\visualize.py
voice_mngt\pitch.py --the rest (see line 161)
testing\simulation/results.py
testinf\simulation/simulation.py
testing\infer_test.py
testing\spectogram_pipeline.py
testing\voice_pipeline.py

Both Developers :
config.py
tacet\gate.py

Agentic Frameworks(ClaudeCode):
voice_mngt\spectogram.py
voice_mngt\pitch.py --Only track_pitch()
frontend\web\index.html
frontend\web\style.css
...static parts from the asr & tts modules
research for text commits in the ui
generation of a part of the training dataset
---