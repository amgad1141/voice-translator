"""
Pre-warm model files into Windows file cache at login.
Reads model files into RAM so the app loads faster later.
Uses no GPU, exits immediately after reading. Windows reclaims
the RAM if other apps need it — zero permanent resource usage.

This script runs silently (.pyw = no console window).
"""
import os
import glob

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(APP_DIR, "models")

def prewarm():
    for f in glob.glob(os.path.join(MODELS_DIR, "**", "*"), recursive=True):
        if os.path.isfile(f):
            try:
                with open(f, "rb") as fh:
                    while fh.read(8 * 1024 * 1024):  # read in 8MB chunks
                        pass
            except:
                pass

if __name__ == "__main__":
    prewarm()
