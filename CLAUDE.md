# Voice Translator

Arabic-to-English desktop voice translator using Faster-Whisper + CoEdit grammar polishing.

## Repository

- **GitHub:** https://github.com/amgad1141/voice-translator
- **Branch:** `main`
- **Local path:** `L:\Ai applications\whisper\`

## Architecture

| Component | Tech | Purpose |
|-----------|------|---------|
| Speech-to-text | Faster-Whisper (CTranslate2) | Arabic → English translation on GPU |
| Grammar polish | CoEdit-large (grammarly/coedit-large) | Optional grammar correction, GPU with CPU fallback on OOM |
| UI | CustomTkinter | Dark-themed compact desktop app |
| GPU acceleration | PyTorch + CUDA | float16 on GTX 10-series+, int8 on older GPUs |
| Setup automation | PowerShell (setup.ps1) | One-click install on any fresh Windows + NVIDIA PC |

## File Map

```
whisper/
├── whisper_recorder.py   # Main app — UI, recording, translation, history
├── setup.bat             # Entry point: double-click to run setup.ps1
├── setup.ps1             # Full setup: Python, packages, models, shortcut, prewarm
├── launch.bat            # App launcher (pythonw, no console window)
├── prewarm.pyw           # Login-time cache warmer (Task Scheduler)
├── requirements.txt      # pip dependencies (torch+CUDA, faster-whisper, transformers, etc.)
├── SETUP_GUIDE.txt       # Manual setup reference
├── README.md             # GitHub project page
├── .gitignore            # Excludes models/, data/, ffmpeg*, gpu_config.json
├── gpu_config.json       # [generated] GPU settings from setup.ps1
├── models/               # [gitignored] ~5 GB AI models (auto-downloaded)
├── data/                 # [gitignored] translation_history.json
└── ffmpeg*/              # [gitignored] ffmpeg binaries
```

## Critical Rules

### 1. Keep setup files in sync
**Every change to the app MUST also update the setup files.** This is non-negotiable.

When modifying `whisper_recorder.py`, always check and update:
- `requirements.txt` — new pip packages?
- `setup.ps1` — new setup steps, model downloads, config?
- `SETUP_GUIDE.txt` — new prerequisites or folder changes?
- `gpu_config.json` schema — new GPU-related settings?
- `README.md` — new features to document?

**Why:** The user relies on `setup.bat` to reproduce the full environment on a fresh PC. If setup files are out of sync, the fresh install breaks.

### 2. Git workflow after changes
**After EVERY code change, automatically provide the git commands.** Do not wait for the user to ask. Do not just "remind" — output the ready-to-paste block immediately after saving files:
```powershell
cd "L:\Ai applications\whisper"
git add .
git commit -m "description of changes"
git push
```
The commit message must accurately describe what changed. This is mandatory, not optional.

### 3. Do NOT set XDG_CACHE_HOME
Setting the `XDG_CACHE_HOME` environment variable breaks CTranslate2 CUDA performance, causing translation to fall back to CPU (100x slower). Only use `HF_HOME`, `HUGGINGFACE_HUB_CACHE`, and `TRANSFORMERS_CACHE`.

### 4. GPU compatibility
The app must work on ANY NVIDIA GPU, not just RTX 2060 Super:
- `gpu_config.json` stores detected settings (created by setup.ps1)
- `detect_gpu_settings()` in the app auto-detects at runtime if config is missing
- Compute capability ≥ 6.0 (GTX 10-series+) → `float16`
- Compute capability < 6.0 (older) → `int8`
- VRAM ≥ 6 GB → `large-v3`, 4-6 GB → `medium`, < 4 GB → `small`

### 5. No background resource usage
The app must use ZERO GPU/RAM when closed. No system tray, no persistent processes. The `prewarm.pyw` task runs once at login, reads files into OS cache, then exits — no permanent resource usage.

### 6. Grammar polish is optional
CoEdit sometimes makes output worse (removes natural speech flow). Always keep it as a toggle, never force it on. GPU-first with CPU fallback on VRAM overflow.

## Known Issues & Past Bugs

- **0.5s silence prepended to audio** — Whisper cuts the first ~2 seconds of speech. The silence padding is the fix; do not remove it.
- **VRAM limit on 8GB GPUs** — Faster-Whisper large-v3 (~3GB) + CoEdit (~1.2GB) can exceed VRAM under load. The OOM try/except in `GrammarPolisher` handles this by falling back to CPU.
- **Regular Whisper (openai-whisper) is 100x slower** — user only wants Faster-Whisper. The Whisper engine code is kept for compatibility but is not the primary engine.
- **Translation history capped at 20 entries** — auto-pruned in `TranslationHistory.add()`.

## User Preferences

- Always respond in English, even when the user writes in Arabic
- Prefer automated one-click solutions over multi-step manual processes
- No quality loss — don't trade accuracy for speed
- Keep the UI compact — everything visible without scrolling
- Auto-copy translations to clipboard
