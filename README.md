# 🎙 Voice Translator

A desktop app that translates Arabic speech to English text in real-time using AI, with optional grammar polishing.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![CUDA](https://img.shields.io/badge/NVIDIA-CUDA-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Features

- **Real-time Arabic → English translation** using Faster-Whisper (CTranslate2)
- **Grammar polishing** with CoEdit-large (optional, toggle on/off)
- **GPU auto-detection** — adapts model size and precision to your NVIDIA GPU
- **Auto-copy to clipboard** after every translation
- **Translation history** — view, copy, and export your last 20 translations
- **One-click setup** — works on a fresh Windows PC with just an NVIDIA GPU
- **Zero background usage** — no GPU/RAM usage when the app is closed

## Requirements

- **Windows 10/11**
- **NVIDIA GPU** (any GTX 10-series or newer) with drivers installed
- **Internet** (for first-time model download, ~5 GB)

## Quick Setup

1. Download or clone this repo
2. Double-click **`setup.bat`**
3. Wait for setup to finish (installs Python, packages, downloads models)
4. Double-click **"Voice Translator"** on your Desktop

That's it. The setup script handles everything automatically, even on a completely fresh Windows install.

## How It Works

1. Click **Record** and speak in Arabic
2. Click **Stop** — the app translates your speech to English
3. The translation is **auto-copied** to your clipboard
4. Toggle **Grammar** on/off for polished output

## GPU Compatibility

| GPU | VRAM | Model | Speed |
|-----|------|-------|-------|
| GTX 1050 | 2 GB | small + int8 | Good |
| GTX 1060 | 6 GB | large-v3 + float16 | Fast |
| RTX 2060+ | 8 GB | large-v3 + float16 | Fast |
| RTX 3070+ | 8+ GB | large-v3 + float16 | Fast |
| RTX 4090 | 24 GB | large-v3 + float16 | Fast |

The app automatically detects your GPU and picks the best settings.

## Project Structure

```
whisper/
├── whisper_recorder.py    # Main application
├── setup.bat              # One-click setup launcher
├── setup.ps1              # Full setup script
├── launch.bat             # App launcher
├── requirements.txt       # Python dependencies
├── prewarm.pyw            # Startup cache warmer (optional)
├── .gitignore             # Git ignore rules
├── SETUP_GUIDE.txt        # Manual setup reference
├── models/                # AI models (~5 GB, auto-downloaded)
│   └── huggingface/hub/
└── data/                  # Translation history
    └── translation_history.json
```

## Tech Stack

- **[Faster-Whisper](https://github.com/SYSTRAN/faster-whisper)** — CTranslate2-based speech recognition
- **[CoEdit-large](https://huggingface.co/grammarly/coedit-large)** — Grammar correction model by Grammarly
- **[CustomTkinter](https://github.com/TomSchimansky/CustomTkinter)** — Modern dark-themed UI
- **PyTorch + CUDA** — GPU acceleration

## Manual Setup

If `setup.bat` doesn't work:

```powershell
# Install Python 3.12
winget install Python.Python.3.12

# Install packages
pip install -r requirements.txt

# Launch
python whisper_recorder.py
```

Models download automatically on first launch.

## License

MIT
