import customtkinter as ctk
from tkinter import messagebox
import threading
import re
import numpy as np
import tempfile
import os
import shutil
import glob
import time
import json
from datetime import datetime

# ── App root (folder where this script lives) ───────────────────────
APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(APP_DIR, "models")
WHISPER_CACHE = os.path.join(MODELS_DIR, "whisper")
HF_CACHE = os.path.join(MODELS_DIR, "huggingface")
DATA_DIR = os.path.join(APP_DIR, "data")

os.makedirs(WHISPER_CACHE, exist_ok=True)
os.makedirs(HF_CACHE, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

os.environ["HF_HOME"] = HF_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(HF_CACHE, "hub")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(HF_CACHE, "hub")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# ── Lazy imports ─────────────────────────────────────────────────────
sd = None
wav = None
whisper = None
WhisperModel = None
AutoTokenizer = None
AutoModelForSeq2SeqLM = None
torch = None

def _import_audio():
    global sd, wav
    import sounddevice; import scipy.io.wavfile
    sd = sounddevice; wav = scipy.io.wavfile

def _import_engines():
    global whisper, WhisperModel, ENGINES
    try:
        import whisper as _w; whisper = _w
        ENGINES["Whisper"] = {"models": ["tiny","base","small","medium","large-v3"]}
    except ImportError: pass
    try:
        from faster_whisper import WhisperModel as _WM
        global WhisperModel; WhisperModel = _WM
        ENGINES["Faster-Whisper"] = {"models": ["tiny","base","small","medium","large-v3"]}
    except ImportError: pass

def _import_polish():
    global AutoTokenizer, AutoModelForSeq2SeqLM, torch, POLISH_AVAILABLE
    try:
        from transformers import AutoTokenizer as _AT, AutoModelForSeq2SeqLM as _AM
        import torch as _t
        AutoTokenizer = _AT; AutoModelForSeq2SeqLM = _AM; torch = _t
        POLISH_AVAILABLE = True
    except ImportError: pass

# ── Theme ────────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

BG       = "#0f0f1a"
BG_CARD  = "#1a1a2e"
BG_INPUT = "#16213e"
RED      = "#e94560"
GREEN    = "#00d474"
BLUE     = "#0f3460"
ORANGE   = "#f59e0b"
WHITE    = "#ffffff"
GRAY     = "#a0a0b0"
DIM      = "#6a6a7a"

HISTORY_FILE = os.path.join(DATA_DIR, "translation_history.json")
GPU_CONFIG_FILE = os.path.join(APP_DIR, "gpu_config.json")
ENGINES = {}
POLISH_AVAILABLE = False
COEDIT_MODEL_ID = "grammarly/coedit-large"
COEDIT_CACHE_NAME = "models--grammarly--coedit-large"


# ── GPU auto-detection ──────────────────────────────────────────
def detect_gpu_settings():
    """Detect GPU capabilities and return (compute_type, recommended_model).
    Works on any NVIDIA GPU: GTX 10-series and newer use float16,
    older GPUs fall back to int8. Model size adapts to VRAM."""

    # Try reading saved config first (created by setup.ps1)
    if os.path.exists(GPU_CONFIG_FILE):
        try:
            with open(GPU_CONFIG_FILE, 'r') as f:
                cfg = json.load(f)
            return cfg.get("compute_type", "float16"), cfg.get("recommended_model", "large-v3")
        except: pass

    # Auto-detect from GPU at runtime
    compute_type = "float16"
    recommended_model = "large-v3"

    try:
        if torch and torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            vram_gb = props.total_mem / (1024**3)
            cc_major = props.major  # compute capability major version

            # float16 requires compute capability >= 6.0 (Pascal / GTX 10-series)
            if cc_major < 6:
                compute_type = "int8"

            # Adjust model to VRAM
            if vram_gb < 4:
                recommended_model = "small"
            elif vram_gb < 6:
                recommended_model = "medium"
            else:
                recommended_model = "large-v3"
    except: pass

    return compute_type, recommended_model

# Will be set after torch is imported
GPU_COMPUTE_TYPE = "float16"
GPU_RECOMMENDED_MODEL = "large-v3"


# ── Grammar Polisher ─────────────────────────────────────────────────
class GrammarPolisher:
    FILLERS = [
        r'\b(uhm+|uh+|um+|ah+|hmm+|hm+|er+|erm+)\b',
        r'\b(you know|i mean|like|basically|actually|literally|so+)\b',
        r'\b(well|okay so|right so|yeah so)\b',
    ]

    def __init__(self, status_cb=None):
        self.model = self.tokenizer = self.device = None
        self.loaded = False; self.status_cb = status_cb

    def load(self):
        if self.loaded: return
        if not POLISH_AVAILABLE: raise RuntimeError("transformers not installed")
        if self.status_cb: self.status_cb("Loading grammar model...")
        self.tokenizer = AutoTokenizer.from_pretrained(COEDIT_MODEL_ID)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(COEDIT_MODEL_ID)
        if torch.cuda.is_available():
            try:
                self.model = self.model.to("cuda"); self.device = "cuda"
                torch.cuda.synchronize()
            except (RuntimeError, torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()
                self.model = self.model.to("cpu"); self.device = "cpu"
        else:
            self.device = "cpu"; self.model = self.model.to("cpu")
        self.loaded = True

    def clean_fillers(self, text):
        for p in self.FILLERS: text = re.sub(p, '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        text = re.sub(r'([.,!?;:])\1+', r'\1', text)
        return re.sub(r'^\s*[.,]\s*', '', text).strip()

    def fix_grammar(self, text):
        if not self.loaded: self.load()
        parts = []
        for s in re.split(r'(?<=[.!?])\s+', text):
            if not s.strip(): continue
            try:
                inp = self.tokenizer(f"Fix grammatical errors in this sentence: {s}",
                    return_tensors="pt", max_length=256, truncation=True).to(self.device)
                with torch.no_grad():
                    out = self.model.generate(**inp, max_length=256, num_beams=5, early_stopping=True)
                parts.append(self.tokenizer.decode(out[0], skip_special_tokens=True))
            except (RuntimeError, torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache(); self.model = self.model.to("cpu"); self.device = "cpu"
                inp = self.tokenizer(f"Fix grammatical errors in this sentence: {s}",
                    return_tensors="pt", max_length=256, truncation=True).to("cpu")
                with torch.no_grad():
                    out = self.model.generate(**inp, max_length=256, num_beams=5, early_stopping=True)
                parts.append(self.tokenizer.decode(out[0], skip_special_tokens=True))
        return " ".join(parts)

    def polish(self, text):
        cleaned = self.clean_fillers(text)
        if not cleaned: return text
        p = self.fix_grammar(cleaned)
        return (p[0].upper() + p[1:]) if p else text


# ── Translation History ──────────────────────────────────────────────
class TranslationHistory:
    def __init__(self):
        self.entries = []
        try:
            if os.path.exists(HISTORY_FILE):
                with open(HISTORY_FILE, 'r', encoding='utf-8') as f: self.entries = json.load(f)
        except: self.entries = []

    def save(self):
        os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
        try:
            with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.entries, f, ensure_ascii=False, indent=2)
        except: pass

    def add(self, text, engine, duration, polished=False):
        self.entries.insert(0, {"text": text, "engine": engine, "duration": duration,
            "polished": polished, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
        self.entries = self.entries[:20]
        self.save()

    def clear(self): self.entries = []; self.save()


# ── Model cache helpers ──────────────────────────────────────────────
def get_hf_cache(): return os.path.join(HF_CACHE, "hub")

def get_model_cache_path(engine, model):
    if engine == "Whisper":
        if not os.path.exists(WHISPER_CACHE): return None
        for f in os.listdir(WHISPER_CACHE):
            if model in f: return os.path.join(WHISPER_CACHE, f)
    elif engine == "Faster-Whisper":
        d = get_hf_cache()
        if not os.path.exists(d): return None
        for p in [f"models--*faster-whisper-{model}*", f"models--*whisper-{model}*"]:
            m = glob.glob(os.path.join(d, p))
            if m: return m[0]
    return None

def folder_size(path):
    if os.path.isfile(path): return os.path.getsize(path)
    return sum(os.path.getsize(os.path.join(dp,f)) for dp,_,fns in os.walk(path) for f in fns
               if os.path.exists(os.path.join(dp,f)))

def fmt_size(b):
    if b < 1024**2: return f"{b/1024:.0f} KB"
    elif b < 1024**3: return f"{b/1024**2:.0f} MB"
    return f"{b/1024**3:.1f} GB"

def is_coedit_downloaded():
    d = get_hf_cache(); os.makedirs(d, exist_ok=True)
    t = os.path.join(d, COEDIT_CACHE_NAME)
    return (True, folder_size(t)) if os.path.exists(t) else (False, None)


# ── Engine wrappers ──────────────────────────────────────────────────
class WhisperEngine:
    def __init__(self, model_name, cb):
        cb(f"Loading Whisper {model_name}...")
        self.model = whisper.load_model(model_name, download_root=WHISPER_CACHE)
        self.name = f"Whisper ({model_name})"
    def translate(self, path):
        return self.model.transcribe(path, language="Arabic", task="translate")["text"].strip()

class FasterWhisperEngine:
    def __init__(self, model_name, cb):
        cb(f"Loading Faster-Whisper {model_name}...")
        self.model = WhisperModel(model_name, device="cuda", compute_type=GPU_COMPUTE_TYPE)
        self.name = f"Faster-Whisper ({model_name})"
    def translate(self, path):
        segs, _ = self.model.transcribe(path, language="ar", task="translate")
        return " ".join(s.text for s in segs).strip()

ENGINE_CLASSES = {"Whisper": WhisperEngine, "Faster-Whisper": FasterWhisperEngine}


# ── Waveform ─────────────────────────────────────────────────────────
import tkinter as tk

class WaveformBar(tk.Canvas):
    def __init__(self, parent, width=500, height=40, **kw):
        super().__init__(parent, width=width, height=height, bg=BG, highlightthickness=0, **kw)
        self.n = 40; self.bw = width/self.n*0.6; self.bg = width/self.n*0.4
        self.h_vals = [0.1]*self.n; self.targets = [0.1]*self.n
        self.on = False; self.w = width; self.h = height

    def start(self): self.on = True; self._tick()
    def stop(self): self.on = False; self.h_vals = [0.1]*self.n; self._draw()

    def feed(self, chunk):
        if not len(chunk): return
        cs = max(1, len(chunk)//self.n)
        for i in range(self.n):
            s = i*cs; e = min(s+cs, len(chunk))
            if s < len(chunk): self.targets[i] = max(0.05, min(1.0, np.abs(chunk[s:e]).mean()*8))

    def _tick(self):
        if not self.on: return
        for i in range(self.n):
            self.h_vals[i] += (self.targets[i]-self.h_vals[i])*0.3
            self.targets[i] = max(0.05, self.targets[i]*0.85)
        self._draw(); self.after(50, self._tick)

    def _draw(self):
        self.delete("all"); tw = self.bw+self.bg; sx = (self.w-tw*self.n)/2; my = self.h/2
        for i in range(self.n):
            x = sx+i*tw; bh = self.h_vals[i]*(self.h*0.8)
            r = int(233*(1-self.h_vals[i])); g = int(69*(1-self.h_vals[i])+212*self.h_vals[i])
            b = int(96*(1-self.h_vals[i])+116*self.h_vals[i])
            self.create_rectangle(x, my-bh/2, x+self.bw, my+bh/2, fill=f"#{r:02x}{g:02x}{b:02x}", outline="")


# ── Main App ─────────────────────────────────────────────────────────
class TranslatorApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Voice Translator")
        self.geometry("650x680")
        self.configure(fg_color=BG)
        self.resizable(True, True)
        self.minsize(550, 580)

        self.recording = False; self.audio_data = []; self.sample_rate = 16000
        self.engine = None; self.polisher = None; self.record_start_time = 0
        self.stream = None; self.history = TranslationHistory()
        self.mic_devices = {}

        # ── Header ───────────────────────────────────────────────
        ctk.CTkLabel(self, text="🎙 Voice Translator", font=ctk.CTkFont(size=22, weight="bold"),
                     text_color=WHITE, fg_color=BG).pack(pady=(12, 0))

        self.status_label = ctk.CTkLabel(self, text="⏳ Starting...",
            font=ctk.CTkFont(size=12), text_color=ORANGE, fg_color=BG)
        self.status_label.pack(pady=(2, 8))

        # ── Settings Row ─────────────────────────────────────────
        settings = ctk.CTkFrame(self, fg_color=BG_CARD, corner_radius=12)
        settings.pack(fill="x", padx=15, pady=(0, 6))

        row1 = ctk.CTkFrame(settings, fg_color="transparent")
        row1.pack(fill="x", padx=12, pady=(10, 4))

        ctk.CTkLabel(row1, text="Engine", font=ctk.CTkFont(size=11), text_color=DIM, width=50).pack(side="left")
        self.engine_var = ctk.StringVar(value="Faster-Whisper")
        self.engine_dd = ctk.CTkOptionMenu(row1, variable=self.engine_var, values=["Loading..."],
            width=150, height=28, font=ctk.CTkFont(size=11), fg_color=BLUE, button_color=BLUE,
            command=self.on_engine_change)
        self.engine_dd.pack(side="left", padx=(4, 10))

        ctk.CTkLabel(row1, text="Model", font=ctk.CTkFont(size=11), text_color=DIM, width=45).pack(side="left")
        self.model_var = ctk.StringVar(value="large-v3")
        self.model_dd = ctk.CTkOptionMenu(row1, variable=self.model_var, values=["Loading..."],
            width=120, height=28, font=ctk.CTkFont(size=11), fg_color=BLUE, button_color=BLUE)
        self.model_dd.pack(side="left", padx=(4, 10))

        self.polish_var = ctk.BooleanVar(value=False)
        self.polish_sw = ctk.CTkSwitch(row1, text="Grammar", variable=self.polish_var,
            width=40, height=20, font=ctk.CTkFont(size=11), text_color=GRAY,
            progress_color=GREEN, button_color=WHITE, state="disabled")
        self.polish_sw.pack(side="right")

        row2 = ctk.CTkFrame(settings, fg_color="transparent")
        row2.pack(fill="x", padx=12, pady=(0, 10))

        ctk.CTkLabel(row2, text="Mic", font=ctk.CTkFont(size=11), text_color=DIM, width=50).pack(side="left")
        self.mic_var = ctk.StringVar(value="Loading...")
        self.mic_dd = ctk.CTkOptionMenu(row2, variable=self.mic_var, values=["Loading..."],
            width=350, height=28, font=ctk.CTkFont(size=11), fg_color=BLUE, button_color=BLUE,
            dynamic_resizing=False)
        self.mic_dd.pack(side="left", padx=(4, 0), fill="x", expand=True)

        # ── Waveform + Record ────────────────────────────────────
        self.waveform = WaveformBar(self, width=580, height=35)
        self.waveform.pack(pady=(6, 2))

        rec_frame = ctk.CTkFrame(self, fg_color=BG)
        rec_frame.pack(pady=(2, 2))

        self.record_btn = ctk.CTkButton(rec_frame, text="🎙  Record", width=220, height=55,
            font=ctk.CTkFont(size=18, weight="bold"), fg_color=RED, hover_color="#c93550",
            corner_radius=28, command=self.toggle_recording, state="disabled")
        self.record_btn.pack(side="left", padx=5)

        self.timer_label = ctk.CTkLabel(rec_frame, text="", font=ctk.CTkFont(size=16, weight="bold"),
            text_color=ORANGE, width=70)
        self.timer_label.pack(side="left", padx=(10, 0))

        # ── Tabbed Output (Translation + History) ─────────────────
        self.tabs = ctk.CTkTabview(self, fg_color=BG_CARD, corner_radius=12,
            segmented_button_fg_color=BLUE, segmented_button_selected_color="#2563eb",
            segmented_button_unselected_color=BG_CARD, segmented_button_selected_hover_color="#1d4ed8")
        self.tabs.pack(fill="both", expand=True, padx=15, pady=(6, 8))

        # ── Translation tab ──────────────────────────────────────
        trans_tab = self.tabs.add("📝 Translation")

        trans_top = ctk.CTkFrame(trans_tab, fg_color="transparent")
        trans_top.pack(fill="x", padx=4, pady=(4, 2))

        self.autocopy_label = ctk.CTkLabel(trans_top, text="", font=ctk.CTkFont(size=11), text_color=GREEN)
        self.autocopy_label.pack(side="right")

        self.text_box = ctk.CTkTextbox(trans_tab, font=ctk.CTkFont(size=14), height=80,
            fg_color=BG_INPUT, text_color=WHITE, border_width=1, border_color=BLUE,
            corner_radius=8, wrap="word")
        self.text_box.pack(fill="both", expand=True, padx=4, pady=(0, 4))

        btn_row = ctk.CTkFrame(trans_tab, fg_color="transparent")
        btn_row.pack(pady=(0, 4))

        ctk.CTkButton(btn_row, text="📋 Copy", width=90, height=28,
            font=ctk.CTkFont(size=11, weight="bold"), fg_color="#2563eb", hover_color="#1d4ed8",
            corner_radius=8, command=self.copy_text).pack(side="left", padx=3)

        ctk.CTkButton(btn_row, text="🗑 Clear", width=70, height=28,
            font=ctk.CTkFont(size=11, weight="bold"), fg_color="#374151", hover_color="#4b5563",
            corner_radius=8, command=self.clear_text).pack(side="left", padx=3)

        # ── History tab ──────────────────────────────────────────
        hist_tab = self.tabs.add("📜 History")

        hist_top = ctk.CTkFrame(hist_tab, fg_color="transparent")
        hist_top.pack(fill="x", padx=4, pady=(4, 2))

        ctk.CTkButton(hist_top, text="💾 Export", width=70, height=24,
            font=ctk.CTkFont(size=10, weight="bold"), fg_color=BLUE, hover_color="#1a4a8a",
            corner_radius=6, command=self.export_history).pack(side="right", padx=2)

        ctk.CTkButton(hist_top, text="🗑 Clear All", width=80, height=24,
            font=ctk.CTkFont(size=10, weight="bold"), fg_color="#374151", hover_color="#4b5563",
            corner_radius=6, command=self.clear_history).pack(side="right", padx=2)

        self.history_count_label = ctk.CTkLabel(hist_top, text="",
            font=ctk.CTkFont(size=10), text_color=DIM)
        self.history_count_label.pack(side="left")

        self.history_scroll = ctk.CTkScrollableFrame(hist_tab, fg_color="transparent",
            corner_radius=8)
        self.history_scroll.pack(fill="both", expand=True, padx=4, pady=(0, 4))

        self.history_widgets = []
        self.refresh_history_display()

        # ── Start background loading ─────────────────────────────
        threading.Thread(target=self._bg_init, daemon=True).start()

    # ── Background init ──────────────────────────────────────────
    def _bg_init(self):
        try:
            self.set_status("⏳ Loading libraries...")
            _import_audio()
            self.after(0, self.refresh_mics)

            self.set_status("⏳ Loading Faster-Whisper...")
            _import_engines()
            self.after(0, self._setup_engines)

            # Detect GPU capabilities (needs torch imported first)
            global GPU_COMPUTE_TYPE, GPU_RECOMMENDED_MODEL
            _import_polish()  # imports torch
            GPU_COMPUTE_TYPE, GPU_RECOMMENDED_MODEL = detect_gpu_settings()

            model = GPU_RECOMMENDED_MODEL
            if "Faster-Whisper" in ENGINES:
                self.set_status(f"⏳ Loading model into GPU ({GPU_COMPUTE_TYPE})...")
                self.engine = FasterWhisperEngine(model, self.set_status)
                self.set_status(f"✓ Ready — {self.engine.name}")
                self.after(0, lambda: self.record_btn.configure(state="normal"))
                self.after(0, lambda m=model: self.model_var.set(m))
            elif "Whisper" in ENGINES:
                self.set_status("⏳ Loading model into GPU...")
                self.engine = WhisperEngine(model, self.set_status)
                self.set_status(f"✓ Ready — {self.engine.name}")
                self.after(0, lambda: self.record_btn.configure(state="normal"))
                self.after(0, lambda m=model: self.model_var.set(m))

            threading.Thread(target=self._bg_grammar, daemon=True).start()
        except Exception as e:
            self.set_status(f"Error: {e}")

    def _bg_grammar(self):
        if not POLISH_AVAILABLE:
            _import_polish()
        if POLISH_AVAILABLE:
            self.after(0, lambda: self.polish_sw.configure(state="normal"))
            self.polisher = GrammarPolisher(self.set_status)
            self.polisher.load()
            self.after(0, lambda: self.polish_var.set(True))
            engine_name = self.engine.name if self.engine else "Ready"
            self.set_status(f"✓ {engine_name} — Grammar ON")

    def _setup_engines(self):
        names = list(ENGINES.keys())
        if names:
            self.engine_dd.configure(values=names)
            self.engine_var.set("Faster-Whisper" if "Faster-Whisper" in names else names[0])
            self.on_engine_change()

    # ── Mic ──────────────────────────────────────────────────────
    def refresh_mics(self):
        if not sd: return
        devs = sd.query_devices(); self.mic_devices = {}; names = []
        default = sd.default.device[0]
        skip = ["sound mapper","primary sound","stereo mix","line in","oculusvad",
                "virtual","loopback","wave speaker","bthhfenum","hands-free ag"]
        apis = [h['name'] for h in sd.query_hostapis()]
        for i, d in enumerate(devs):
            if d['max_input_channels'] <= 0: continue
            nl = d['name'].lower()
            if any(k in nl for k in skip): continue
            api = apis[d['hostapi']] if d['hostapi'] < len(apis) else ""
            if "wasapi" in api.lower(): continue
            n = d['name']
            if n in self.mic_devices: n = f"{n} ({i})"
            self.mic_devices[n] = i; names.append(n)
        if names:
            self.mic_dd.configure(values=names)
            dn = next((n for n,i in self.mic_devices.items() if i==default), names[0])
            self.mic_var.set(dn)

    # ── Engine change ────────────────────────────────────────────
    def on_engine_change(self, *_):
        e = self.engine_var.get()
        if e in ENGINES:
            self.model_dd.configure(values=ENGINES[e]["models"])
            self.model_var.set("large-v3")

    # ── Recording ────────────────────────────────────────────────
    def toggle_recording(self):
        if not self.recording: self.start_rec()
        else: self.stop_rec()

    def start_rec(self):
        self.recording = True; self.audio_data = []
        self.record_start_time = time.time()
        self.record_btn.configure(text="⏹  Stop", fg_color=GREEN, hover_color="#00b060")
        self.set_status("🔴 Recording... speak now")
        self.waveform.start(); self._tick_timer()

        def cb(indata, frames, ti, status):
            if self.recording:
                self.audio_data.append(indata.copy())
                self.waveform.feed(indata.flatten())

        idx = self.mic_devices.get(self.mic_var.get())
        self.stream = sd.InputStream(samplerate=self.sample_rate, channels=1,
            dtype='float32', callback=cb, blocksize=1024, device=idx)
        self.stream.start()

    def _tick_timer(self):
        if not self.recording: return
        e = int(time.time()-self.record_start_time)
        self.timer_label.configure(text=f"{e//60:02d}:{e%60:02d}")
        self.after(1000, self._tick_timer)

    def stop_rec(self):
        self.recording = False
        if self.stream: self.stream.stop(); self.stream.close()
        self.waveform.stop()
        self.record_btn.configure(text="🎙  Record", fg_color=RED, hover_color="#c93550", state="disabled")
        self.timer_label.configure(text="")
        self.set_status("⏳ Translating...")
        threading.Thread(target=self._translate, daemon=True).start()

    # ── Translation ──────────────────────────────────────────────
    def _translate(self):
        try:
            audio = np.concatenate(self.audio_data, axis=0).flatten()
            silence = np.zeros(int(self.sample_rate * 0.5), dtype=audio.dtype)
            audio = np.concatenate([silence, audio])
            tmp = os.path.join(tempfile.gettempdir(), "whisper_temp.wav")
            wav.write(tmp, self.sample_rate, (audio * 32767).astype(np.int16))

            t0 = time.time()
            text = self.engine.translate(tmp)
            tt = round(time.time()-t0, 1)

            pt = 0
            if self.polish_var.get() and self.polisher and self.polisher.loaded:
                self.set_status("⏳ Polishing grammar...")
                t0 = time.time(); text = self.polisher.polish(text); pt = round(time.time()-t0, 1)

            self.text_box.delete("1.0", "end")
            self.text_box.insert("1.0", text)

            # Auto-copy to clipboard
            self.clipboard_clear(); self.clipboard_append(text)
            self.after(0, lambda: self.autocopy_label.configure(text="✓ Auto-copied!"))
            self.after(3000, lambda: self.autocopy_label.configure(text=""))

            self.history.add(text, self.engine.name, round(tt+pt, 1), pt > 0)
            self.after(0, self.refresh_history_display)

            s = f"✓ {tt}s"
            if pt > 0: s += f" + polish {pt}s"
            self.set_status(s)

        except Exception as e:
            self.set_status(f"Error: {e}")
        finally:
            self.after(0, lambda: self.record_btn.configure(state="normal"))
            try: os.remove(tmp)
            except: pass

    # ── Helpers ───────────────────────────────────────────────────
    def set_status(self, t): self.status_label.configure(text=t)

    def copy_text(self):
        t = self.text_box.get("1.0", "end").strip()
        if t:
            self.clipboard_clear(); self.clipboard_append(t)
            self.autocopy_label.configure(text="✓ Copied!")
            self.after(2000, lambda: self.autocopy_label.configure(text=""))

    def clear_text(self):
        self.text_box.delete("1.0", "end")
        self.set_status("Cleared")

    # ── History display ─────────────────────────────────────────
    def refresh_history_display(self):
        """Rebuild the history list showing last 10 translations."""
        # Clear old widgets
        for w in self.history_widgets:
            w.destroy()
        self.history_widgets.clear()

        entries = self.history.entries[:10]
        self.history_count_label.configure(
            text=f"{len(self.history.entries)} total" if self.history.entries else "No history yet")

        for i, entry in enumerate(entries):
            # Each history item is a compact card
            item = ctk.CTkFrame(self.history_scroll, fg_color=BG_INPUT, corner_radius=8)
            item.pack(fill="x", pady=(0, 4), padx=2)

            # Top row: timestamp + engine + duration
            top = ctk.CTkFrame(item, fg_color="transparent")
            top.pack(fill="x", padx=8, pady=(6, 0))

            ts = entry.get('timestamp', '')
            # Show only time if today, otherwise show date+time
            ts_short = ts.split(' ')[-1] if ts else ''  # just HH:MM:SS
            ts_date = ts.split(' ')[0] if ts else ''

            meta = f"{ts_short}"
            if entry.get('polished'): meta += " • polished"
            meta += f" • {entry.get('duration', '?')}s"

            ctk.CTkLabel(top, text=ts_date, font=ctk.CTkFont(size=9),
                text_color=DIM).pack(side="left")
            ctk.CTkLabel(top, text=meta, font=ctk.CTkFont(size=9),
                text_color=DIM).pack(side="right")

            # Translation text (truncated to 2 lines visually)
            text = entry.get('text', '')
            display_text = text[:150] + "..." if len(text) > 150 else text

            text_label = ctk.CTkLabel(item, text=display_text,
                font=ctk.CTkFont(size=11), text_color=GRAY,
                wraplength=550, justify="left", anchor="w")
            text_label.pack(fill="x", padx=8, pady=(2, 2))

            # Copy button for this entry
            copy_btn = ctk.CTkButton(item, text="📋", width=30, height=20,
                font=ctk.CTkFont(size=10), fg_color="transparent", hover_color=BLUE,
                corner_radius=4, command=lambda t=text: self._copy_history_item(t))
            copy_btn.pack(anchor="e", padx=8, pady=(0, 4))

            self.history_widgets.append(item)

        if not entries:
            empty = ctk.CTkLabel(self.history_scroll, text="No translations yet.\nRecord something to see history here.",
                font=ctk.CTkFont(size=12), text_color=DIM)
            empty.pack(pady=30)
            self.history_widgets.append(empty)

    def _copy_history_item(self, text):
        self.clipboard_clear(); self.clipboard_append(text)
        self.set_status("✓ Copied from history")

    def clear_history(self):
        self.history.clear()
        self.refresh_history_display()
        self.set_status("History cleared")

    def export_history(self):
        if not self.history.entries: self.set_status("No history"); return
        p = os.path.join(os.path.expanduser("~"), "Desktop", "translation_history.txt")
        try:
            with open(p, 'w', encoding='utf-8') as f:
                f.write("Translation History\n" + "="*50 + "\n\n")
                for e in self.history.entries:
                    f.write(f"[{e['timestamp']}] {e['engine']}\n{e['text']}\n{'-'*30}\n\n")
            self.set_status("✓ Exported to Desktop")
        except Exception as e: self.set_status(f"Export error: {e}")


if __name__ == "__main__":
    app = TranslatorApp()
    app.mainloop()
