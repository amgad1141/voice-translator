#Requires -Version 5.1
<#
╔══════════════════════════════════════════════════════════════════╗
║            Voice Translator — One-Click Setup                    ║
║                                                                  ║
║  Works on a COMPLETELY fresh Windows PC.                         ║
║  Only requirement: NVIDIA GPU with drivers installed.            ║
║                                                                  ║
║  What it does:                                                   ║
║    1. Installs Python 3.12 (if not found)                        ║
║    2. Installs ffmpeg (if not found)                             ║
║    3. Detects NVIDIA GPU + VRAM + CUDA capability                ║
║    4. Installs all Python packages (torch, faster-whisper, etc.) ║
║    5. Downloads AI models (~5 GB total)                          ║
║    6. Creates Desktop shortcut                                   ║
║    7. Registers pre-warm task (faster cold starts)               ║
║    8. Runs full verification test                                ║
║                                                                  ║
║  Run: Double-click setup.bat                                     ║
║  Or:  powershell -ExecutionPolicy Bypass -File setup.ps1         ║
╚══════════════════════════════════════════════════════════════════╝
#>

# ── Config ──────────────────────────────────────────────────────
$ErrorActionPreference = "Continue"
$APP_DIR   = Split-Path -Parent $MyInvocation.MyCommand.Path
$MODELS    = Join-Path $APP_DIR "models"
$HF_CACHE  = Join-Path $MODELS "huggingface"
$HF_HUB    = Join-Path $HF_CACHE "hub"
$DATA_DIR  = Join-Path $APP_DIR "data"

$passed = 0
$failed = 0
$warnings = 0

# GPU info (filled in step 2)
$script:GPU_NAME = ""
$script:GPU_VRAM_MB = 0
$script:GPU_COMPUTE_CAP = ""

# ── Helper functions ────────────────────────────────────────────
function Write-Step  { param($msg) Write-Host "`n━━━ $msg ━━━" -ForegroundColor Cyan }
function Write-OK    { param($msg) Write-Host "  ✓ $msg" -ForegroundColor Green; $script:passed++ }
function Write-Fail  { param($msg) Write-Host "  ✗ $msg" -ForegroundColor Red;   $script:failed++ }
function Write-Warn  { param($msg) Write-Host "  ⚠ $msg" -ForegroundColor Yellow; $script:warnings++ }
function Write-Info  { param($msg) Write-Host "  → $msg" -ForegroundColor Gray }

function Test-CommandExists { param($cmd) $null -ne (Get-Command $cmd -ErrorAction SilentlyContinue) }

function Pause-Continue {
    Write-Host "`nPress any key to continue..." -ForegroundColor DarkGray
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
}

function Refresh-Path {
    # Reload PATH from registry so newly installed programs are found
    $machinePath = [Environment]::GetEnvironmentVariable("PATH", "Machine")
    $userPath    = [Environment]::GetEnvironmentVariable("PATH", "User")
    $env:PATH    = "$machinePath;$userPath"
}

# ── Banner ──────────────────────────────────────────────────────
Clear-Host
Write-Host ""
Write-Host "  ╔════════════════════════════════════╗" -ForegroundColor DarkCyan
Write-Host "  ║   🎙  Voice Translator Setup       ║" -ForegroundColor White
Write-Host "  ╚════════════════════════════════════╝" -ForegroundColor DarkCyan
Write-Host "  App folder: $APP_DIR" -ForegroundColor DarkGray
Write-Host ""

# ═══════════════════════════════════════════════════════════════
# STEP 1: Python — install if missing
# ═══════════════════════════════════════════════════════════════
Write-Step "1/8  Python"

$pythonCmd = $null

function Find-Python {
    # Search for Python in PATH, common install locations, and Windows Store
    $candidates = @("python", "python3", "py")

    # Also check common install paths directly
    $extraPaths = @(
        "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe",
        "C:\Python312\python.exe",
        "C:\Python311\python.exe",
        "C:\Python310\python.exe",
        "$env:ProgramFiles\Python312\python.exe",
        "$env:ProgramFiles\Python311\python.exe"
    )

    foreach ($cmd in $candidates) {
        if (Test-CommandExists $cmd) {
            $ver = & $cmd --version 2>&1
            if ($ver -match "Python\s+(\d+)\.(\d+)") {
                if ([int]$Matches[1] -ge 3 -and [int]$Matches[2] -ge 10) {
                    return $cmd
                }
            }
        }
    }

    foreach ($path in $extraPaths) {
        if (Test-Path $path) {
            $ver = & $path --version 2>&1
            if ($ver -match "Python\s+(\d+)\.(\d+)") {
                if ([int]$Matches[1] -ge 3 -and [int]$Matches[2] -ge 10) {
                    return $path
                }
            }
        }
    }
    return $null
}

$pythonCmd = Find-Python

if ($pythonCmd) {
    $pyVer = & $pythonCmd --version 2>&1
    Write-OK "Found $pyVer"
} else {
    Write-Info "Python 3.10+ not found — installing automatically..."

    # Check if winget is available (built into Windows 11, Win10 with updates)
    if (Test-CommandExists "winget") {
        Write-Info "Installing Python 3.12 via winget (this takes 1-2 minutes)..."
        try {
            $wingetOut = winget install Python.Python.3.12 --accept-package-agreements --accept-source-agreements --scope user 2>&1
            Write-Host "  $($wingetOut | Select-Object -Last 3 | Out-String)" -ForegroundColor DarkGray

            # Refresh PATH and search again
            Refresh-Path
            Start-Sleep -Seconds 2
            $pythonCmd = Find-Python

            if ($pythonCmd) {
                $pyVer = & $pythonCmd --version 2>&1
                Write-OK "Installed $pyVer"
            } else {
                # winget installed it but PATH not updated in this session
                # Try the known default path
                $defaultPy = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
                if (Test-Path $defaultPy) {
                    $pythonCmd = $defaultPy
                    $pyVer = & $pythonCmd --version 2>&1
                    Write-OK "Installed $pyVer (at $defaultPy)"
                } else {
                    Write-Fail "Python installed but not found in PATH"
                    Write-Host "  Please CLOSE this window and run setup.bat again." -ForegroundColor Yellow
                    Write-Host "  (New programs need a fresh terminal to be found)" -ForegroundColor Yellow
                    Pause-Continue; exit 0
                }
            }
        } catch {
            Write-Fail "winget install failed: $_"
            Write-Host ""
            Write-Host "  Please install Python manually:" -ForegroundColor Yellow
            Write-Host "  → https://www.python.org/downloads/" -ForegroundColor White
            Write-Host "  → Check 'Add Python to PATH' during install!" -ForegroundColor Red
            Write-Host "  Then run setup.bat again." -ForegroundColor Yellow
            Pause-Continue; exit 1
        }
    } else {
        # No winget — try downloading Python installer directly
        Write-Info "winget not available — downloading Python installer..."
        $pyInstaller = Join-Path $env:TEMP "python-3.12-setup.exe"
        try {
            $pyUrl = "https://www.python.org/ftp/python/3.12.7/python-3.12.7-amd64.exe"
            Write-Info "Downloading from python.org..."
            [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
            (New-Object Net.WebClient).DownloadFile($pyUrl, $pyInstaller)

            Write-Info "Running installer (this takes 1-2 minutes)..."
            Start-Process -FilePath $pyInstaller -ArgumentList "/quiet", "InstallAllUsers=0", "PrependPath=1", "Include_pip=1" -Wait

            Refresh-Path
            Start-Sleep -Seconds 2
            $pythonCmd = Find-Python

            if ($pythonCmd) {
                $pyVer = & $pythonCmd --version 2>&1
                Write-OK "Installed $pyVer"
            } else {
                Write-Warn "Python installed — please CLOSE this window and run setup.bat again."
                Pause-Continue; exit 0
            }
        } catch {
            Write-Fail "Download failed: $_"
            Write-Host "  Please install Python manually from https://www.python.org/downloads/" -ForegroundColor Yellow
            Write-Host "  Then run setup.bat again." -ForegroundColor Yellow
            Pause-Continue; exit 1
        } finally {
            if (Test-Path $pyInstaller) { Remove-Item $pyInstaller -Force -ErrorAction SilentlyContinue }
        }
    }
}

$pyPath = if (Test-Path $pythonCmd) { $pythonCmd } else { (Get-Command $pythonCmd -ErrorAction SilentlyContinue).Source }
Write-Info "Location: $pyPath"

# Make sure pip is available
Write-Info "Ensuring pip is up to date..."
& $pythonCmd -m ensurepip --upgrade 2>&1 | Out-Null
& $pythonCmd -m pip install --upgrade pip --break-system-packages 2>&1 | Out-Null

# ═══════════════════════════════════════════════════════════════
# STEP 2: NVIDIA GPU detection (detailed)
# ═══════════════════════════════════════════════════════════════
Write-Step "2/8  NVIDIA GPU detection"

$hasCuda = $false

if (Test-CommandExists "nvidia-smi") {
    try {
        # Get GPU name
        $gpuName = (nvidia-smi --query-gpu=name --format=csv,noheader 2>&1).Trim()
        # Get VRAM in MB
        $vramStr = (nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>&1).Trim()
        $vramMB  = [int]$vramStr
        # Get driver version
        $driverVer = (nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>&1).Trim()
        # Get CUDA compute capability
        $ccMajor = (nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>&1).Trim()

        $script:GPU_NAME = $gpuName
        $script:GPU_VRAM_MB = $vramMB
        $script:GPU_COMPUTE_CAP = $ccMajor
        $hasCuda = $true

        Write-OK "GPU: $gpuName"
        Write-Info "VRAM: $vramMB MB ($([math]::Round($vramMB/1024, 1)) GB)"
        Write-Info "Driver: $driverVer"
        Write-Info "Compute capability: $ccMajor"

        # Determine what this GPU can handle
        $vramGB = [math]::Round($vramMB / 1024, 1)
        if ($vramGB -ge 6) {
            Write-OK "GPU has enough VRAM for large-v3 model ($vramGB GB)"
        } elseif ($vramGB -ge 4) {
            Write-Warn "GPU has $vramGB GB VRAM — will use medium model instead of large-v3"
        } else {
            Write-Warn "GPU has only $vramGB GB VRAM — will use small model, translation may be less accurate"
        }

        # Check compute capability for float16 support
        if ($ccMajor -match "^(\d+)") {
            $ccNum = [double]$ccMajor
            if ($ccNum -ge 6.0) {
                Write-OK "GPU supports float16 (fast mode)"
            } else {
                Write-Warn "GPU compute capability $ccMajor < 6.0 — will use int8 instead of float16"
                Write-Info "Translation will work but may be slightly slower"
            }
        }

        # Try enabling persistence mode
        $isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
        if ($isAdmin) {
            nvidia-smi -pm 1 2>&1 | Out-Null
            Write-OK "NVIDIA persistence mode enabled"
        }

    } catch {
        Write-Fail "Could not query GPU details: $_"
    }
} else {
    Write-Fail "NVIDIA GPU not detected (nvidia-smi not found)"
    Write-Host ""
    Write-Host "  This app REQUIRES an NVIDIA GPU. Two possible issues:" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  1. No NVIDIA GPU in this PC" -ForegroundColor White
    Write-Host "     → The app cannot run without one (AMD/Intel GPUs not supported)" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  2. NVIDIA drivers not installed" -ForegroundColor White
    Write-Host "     → Download from: https://www.nvidia.com/drivers" -ForegroundColor Gray
    Write-Host "     → Or run: winget install Nvidia.GeForceExperience" -ForegroundColor Gray
    Write-Host ""

    if (Test-CommandExists "winget") {
        $installDriver = Read-Host "  Try to install NVIDIA GeForce Experience via winget? (y/n)"
        if ($installDriver -eq "y") {
            try {
                winget install Nvidia.GeForceExperience --accept-package-agreements --accept-source-agreements
                Write-Warn "GeForce Experience installed. Please:"
                Write-Host "  1. Open GeForce Experience and let it install GPU drivers" -ForegroundColor Yellow
                Write-Host "  2. Restart your PC" -ForegroundColor Yellow
                Write-Host "  3. Run setup.bat again" -ForegroundColor Yellow
                Pause-Continue; exit 0
            } catch {
                Write-Fail "Install failed. Please install NVIDIA drivers manually."
            }
        }
    }

    $cont = Read-Host "  Continue setup without GPU? (the app will be VERY slow) (y/n)"
    if ($cont -ne "y") { Pause-Continue; exit 1 }
}

# ═══════════════════════════════════════════════════════════════
# STEP 3: ffmpeg — install if missing
# ═══════════════════════════════════════════════════════════════
Write-Step "3/8  ffmpeg"

# Check PATH first, then local folder
$ffmpegOK = $false

if (Test-CommandExists "ffmpeg") {
    Write-OK "ffmpeg found in PATH"
    $ffmpegOK = $true
} else {
    # Check local copy
    $ffmpegLocal = Get-ChildItem -Path $APP_DIR -Filter "ffmpeg.exe" -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($ffmpegLocal) {
        $ffmpegBin = $ffmpegLocal.DirectoryName
        $env:PATH = "$ffmpegBin;$env:PATH"

        # Add to user PATH permanently
        $userPath = [Environment]::GetEnvironmentVariable("PATH", "User")
        if ($userPath -notlike "*$ffmpegBin*") {
            [Environment]::SetEnvironmentVariable("PATH", "$userPath;$ffmpegBin", "User")
        }
        Write-OK "ffmpeg found in app folder and added to PATH"
        $ffmpegOK = $true
    }
}

if (-not $ffmpegOK) {
    Write-Info "ffmpeg not found — installing..."

    if (Test-CommandExists "winget") {
        try {
            winget install Gyan.FFmpeg --accept-package-agreements --accept-source-agreements 2>&1 | Out-Null
            Refresh-Path
            Write-OK "ffmpeg installed via winget"
        } catch {
            Write-Warn "winget install failed — downloading ffmpeg directly..."
        }
    }

    # If still not found, download manually
    if (-not (Test-CommandExists "ffmpeg")) {
        try {
            $ffZip = Join-Path $env:TEMP "ffmpeg.zip"
            $ffDir = Join-Path $APP_DIR "ffmpeg"
            $ffUrl = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
            Write-Info "Downloading ffmpeg (~80 MB)..."
            [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
            (New-Object Net.WebClient).DownloadFile($ffUrl, $ffZip)
            Write-Info "Extracting..."
            Expand-Archive -Path $ffZip -DestinationPath $APP_DIR -Force
            Remove-Item $ffZip -Force -ErrorAction SilentlyContinue

            # Find the extracted bin folder
            $ffBin = Get-ChildItem -Path $APP_DIR -Filter "ffmpeg.exe" -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
            if ($ffBin) {
                $ffmpegBin = $ffBin.DirectoryName
                $env:PATH = "$ffmpegBin;$env:PATH"
                $userPath = [Environment]::GetEnvironmentVariable("PATH", "User")
                if ($userPath -notlike "*$ffmpegBin*") {
                    [Environment]::SetEnvironmentVariable("PATH", "$userPath;$ffmpegBin", "User")
                }
                Write-OK "ffmpeg downloaded and added to PATH"
            }
        } catch {
            Write-Fail "Could not install ffmpeg: $_"
            Write-Host "  Download manually: https://ffmpeg.org/download.html" -ForegroundColor Yellow
        }
    }
}

# ═══════════════════════════════════════════════════════════════
# STEP 4: Create folders
# ═══════════════════════════════════════════════════════════════
Write-Step "4/8  Folder structure"

foreach ($dir in @($MODELS, $HF_CACHE, $HF_HUB, $DATA_DIR)) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}
Write-OK "All folders ready"

# ═══════════════════════════════════════════════════════════════
# STEP 5: Install Python packages
# ═══════════════════════════════════════════════════════════════
Write-Step "5/8  Python packages"

$reqFile = Join-Path $APP_DIR "requirements.txt"
if (-not (Test-Path $reqFile)) {
    Write-Fail "requirements.txt not found!"
    Pause-Continue; exit 1
}

Write-Info "Installing packages (first time: ~3 GB download for PyTorch+CUDA)..."
Write-Info "This can take 5-15 minutes depending on internet speed."
Write-Host ""

try {
    & $pythonCmd -m pip install -r $reqFile --break-system-packages 2>&1 | ForEach-Object {
        $line = $_.ToString()
        if ($line -match "^(Collecting|Downloading|Installing|Successfully|Requirement)") {
            Write-Host "  $line" -ForegroundColor DarkGray
        }
    }
    Write-OK "All Python packages installed"
} catch {
    Write-Fail "Package installation had errors: $_"
    Write-Host "  Try manually: $pythonCmd -m pip install -r `"$reqFile`" --break-system-packages" -ForegroundColor Yellow
}

# ═══════════════════════════════════════════════════════════════
# STEP 6: Download AI models
# ═══════════════════════════════════════════════════════════════
Write-Step "6/8  AI models"

$env:HF_HOME = $HF_CACHE
$env:HUGGINGFACE_HUB_CACHE = $HF_HUB
$env:TRANSFORMERS_CACHE = $HF_HUB
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

# Determine best model based on VRAM
$vramGB = [math]::Round($script:GPU_VRAM_MB / 1024, 1)
if ($vramGB -ge 6 -or $script:GPU_VRAM_MB -eq 0) {
    $whisperModel = "large-v3"
    $modelSize = "~3 GB"
} elseif ($vramGB -ge 4) {
    $whisperModel = "medium"
    $modelSize = "~1.5 GB"
} else {
    $whisperModel = "small"
    $modelSize = "~500 MB"
}

if ($script:GPU_VRAM_MB -gt 0) {
    Write-Info "Your GPU has $vramGB GB VRAM → using '$whisperModel' model"
}

# Download Faster-Whisper model
$fwExists = $false
if (Test-Path $HF_HUB) {
    $fwExists = (Get-ChildItem -Path $HF_HUB -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match "whisper" -and $_.Name -match $whisperModel.Replace("-","") }).Count -gt 0
}

if ($fwExists) {
    Write-OK "Faster-Whisper $whisperModel already downloaded"
} else {
    Write-Info "Downloading Faster-Whisper $whisperModel ($modelSize)..."
    try {
        $dlScript = @"
import os, sys
os.environ['HF_HOME'] = r'$HF_CACHE'
os.environ['HUGGINGFACE_HUB_CACHE'] = r'$HF_HUB'
os.environ['TRANSFORMERS_CACHE'] = r'$HF_HUB'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
from faster_whisper import WhisperModel
print('Downloading Faster-Whisper $whisperModel...')
m = WhisperModel('$whisperModel', device='cpu', compute_type='int8')
print('DONE')
del m
"@
        $dlScript | & $pythonCmd - 2>&1 | ForEach-Object {
            $l = $_.ToString()
            if ($l -match "DONE") { Write-OK "Faster-Whisper $whisperModel downloaded" }
            else { Write-Host "  $l" -ForegroundColor DarkGray }
        }
    } catch {
        Write-Warn "Download failed — app will download on first launch"
    }
}

# Download CoEdit grammar model
$coeditPath = Join-Path $HF_HUB "models--grammarly--coedit-large"
if (Test-Path $coeditPath) {
    Write-OK "Grammar model (CoEdit-large) already downloaded"
} else {
    Write-Info "Downloading grammar model (~1.2 GB)..."
    try {
        $gramScript = @"
import os
os.environ['HF_HOME'] = r'$HF_CACHE'
os.environ['HUGGINGFACE_HUB_CACHE'] = r'$HF_HUB'
os.environ['TRANSFORMERS_CACHE'] = r'$HF_HUB'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
print('Downloading grammar model...')
AutoTokenizer.from_pretrained('grammarly/coedit-large')
AutoModelForSeq2SeqLM.from_pretrained('grammarly/coedit-large')
print('DONE')
"@
        $gramScript | & $pythonCmd - 2>&1 | ForEach-Object {
            $l = $_.ToString()
            if ($l -match "DONE") { Write-OK "Grammar model downloaded" }
            else { Write-Host "  $l" -ForegroundColor DarkGray }
        }
    } catch {
        Write-Warn "Download failed — app will download on first launch"
    }
}

# ═══════════════════════════════════════════════════════════════
# STEP 7: Save GPU config + Desktop shortcut + Prewarm
# ═══════════════════════════════════════════════════════════════
Write-Step "7/8  Shortcuts and configuration"

# Save GPU config file so the app knows what settings to use
$gpuConfigPath = Join-Path $APP_DIR "gpu_config.json"
$computeType = "float16"
if ($script:GPU_COMPUTE_CAP -match "^(\d+)") {
    if ([double]$script:GPU_COMPUTE_CAP -lt 6.0) { $computeType = "int8" }
}
# Determine best default model
$defaultModel = "large-v3"
if ($vramGB -gt 0 -and $vramGB -lt 6) { $defaultModel = "medium" }
if ($vramGB -gt 0 -and $vramGB -lt 4) { $defaultModel = "small" }

$gpuConfig = @"
{
    "gpu_name": "$($script:GPU_NAME)",
    "vram_mb": $($script:GPU_VRAM_MB),
    "compute_capability": "$($script:GPU_COMPUTE_CAP)",
    "compute_type": "$computeType",
    "recommended_model": "$defaultModel",
    "setup_date": "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
}
"@
$gpuConfig | Set-Content -Path $gpuConfigPath -Encoding UTF8
Write-OK "GPU config saved (compute_type=$computeType, model=$defaultModel)"

# Desktop shortcut
$desktop = [Environment]::GetFolderPath("Desktop")
$shortcutPath = Join-Path $desktop "Voice Translator.lnk"

try {
    $shell = New-Object -ComObject WScript.Shell
    $sc = $shell.CreateShortcut($shortcutPath)
    $sc.TargetPath = Join-Path $APP_DIR "launch.bat"
    $sc.WorkingDirectory = $APP_DIR
    $sc.WindowStyle = 7  # Minimized
    $sc.Description = "Voice Translator"

    $pythonExe = if (Test-Path $pyPath) { $pyPath } else { (Get-Command $pythonCmd -ErrorAction SilentlyContinue).Source }
    if ($pythonExe) { $sc.IconLocation = "$pythonExe,0" }

    $sc.Save()
    Write-OK "Desktop shortcut created"
} catch {
    Write-Warn "Could not create shortcut: $_"
}

# Prewarm scheduled task
$prewarmScript = Join-Path $APP_DIR "prewarm.pyw"
if (Test-Path $prewarmScript) {
    try {
        $existingTask = Get-ScheduledTask -TaskName "WhisperPrewarm" -ErrorAction SilentlyContinue
        if ($existingTask) {
            Write-OK "Prewarm startup task already registered"
        } else {
            $isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
            if ($isAdmin) {
                $pythonwPath = Join-Path (Split-Path $pyPath) "pythonw.exe"
                if (-not (Test-Path $pythonwPath)) { $pythonwPath = $pyPath }
                $trigger  = New-ScheduledTaskTrigger -AtLogOn
                $action   = New-ScheduledTaskAction -Execute $pythonwPath -Argument "`"$prewarmScript`""
                $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable
                Register-ScheduledTask -TaskName "WhisperPrewarm" -Trigger $trigger -Action $action -Settings $settings -Description "Pre-warm Voice Translator models" -Force | Out-Null
                Write-OK "Prewarm task registered (faster app startup after login)"
            } else {
                Write-Warn "Run setup.bat as Administrator to register prewarm task (optional)"
            }
        }
    } catch {
        Write-Warn "Prewarm task registration failed (non-critical): $_"
    }
}

# ═══════════════════════════════════════════════════════════════
# STEP 8: Full verification
# ═══════════════════════════════════════════════════════════════
Write-Step "8/8  Verification"

$testScript = @"
import sys, os, json
os.environ['HF_HOME'] = r'$HF_CACHE'
os.environ['HUGGINGFACE_HUB_CACHE'] = r'$HF_HUB'
os.environ['TRANSFORMERS_CACHE'] = r'$HF_HUB'
results = []

# 1. Core packages
try:
    import customtkinter, numpy, scipy, sounddevice
    results.append(('Core packages (UI, audio, numpy)', True, ''))
except ImportError as e:
    results.append(('Core packages', False, str(e)))

# 2. PyTorch + CUDA
cuda = False
gpu_name = ''
try:
    import torch
    cuda = torch.cuda.is_available()
    if cuda:
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_mem / 1024**3
        cc = torch.cuda.get_device_properties(0).major
        results.append(('PyTorch CUDA', True, f'{gpu_name} — {vram:.1f} GB VRAM, compute {cc}.x'))
    else:
        results.append(('PyTorch CUDA', False, 'CUDA not available — will be very slow'))
except ImportError as e:
    results.append(('PyTorch', False, str(e)))

# 3. CTranslate2 compute types
try:
    import ctranslate2
    if cuda:
        types = ctranslate2.get_supported_compute_types('cuda')
        has_fp16 = 'float16' in types
        results.append(('CTranslate2 CUDA', True, f'float16={has_fp16}, types={types}'))
    else:
        results.append(('CTranslate2', True, 'CPU only'))
except ImportError as e:
    results.append(('CTranslate2', False, str(e)))

# 4. Faster-Whisper
try:
    from faster_whisper import WhisperModel
    results.append(('Faster-Whisper', True, ''))
except ImportError as e:
    results.append(('Faster-Whisper', False, str(e)))

# 5. Transformers (grammar)
try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    results.append(('Transformers (grammar)', True, ''))
except ImportError as e:
    results.append(('Transformers', False, str(e)))

# 6. ffmpeg
import shutil
ff = shutil.which('ffmpeg')
results.append(('ffmpeg', ff is not None, ff or 'NOT in PATH'))

# 7. Model files
hf_hub = r'$HF_HUB'
fw_found = False
fw_model = ''
if os.path.exists(hf_hub):
    for d in os.listdir(hf_hub):
        dl = d.lower()
        if 'whisper' in dl:
            fw_found = True
            if 'large' in dl: fw_model = 'large-v3'
            elif 'medium' in dl: fw_model = 'medium'
            elif 'small' in dl: fw_model = 'small'
results.append(('Whisper model files', fw_found, fw_model if fw_found else 'NOT downloaded'))

ce_found = os.path.exists(os.path.join(hf_hub, 'models--grammarly--coedit-large'))
results.append(('Grammar model files', ce_found, 'downloaded' if ce_found else 'NOT downloaded'))

# 8. GPU config file
cfg_path = os.path.join(r'$APP_DIR', 'gpu_config.json')
cfg_ok = os.path.exists(cfg_path)
results.append(('GPU config file', cfg_ok, 'ready' if cfg_ok else 'missing'))

for name, ok, detail in results:
    status = 'PASS' if ok else 'FAIL'
    d = f' ({detail})' if detail else ''
    print(f'{status}|{name}{d}')

sys.exit(0 if all(ok for _, ok, _ in results) else 1)
"@

try {
    $testResults = $testScript | & $pythonCmd - 2>&1
    foreach ($line in $testResults) {
        $l = $line.ToString()
        if ($l -match "^PASS\|(.+)") {
            Write-OK $Matches[1]
        } elseif ($l -match "^FAIL\|(.+)") {
            Write-Fail $Matches[1]
        } elseif ($l.Trim()) {
            Write-Info "$l"
        }
    }
} catch {
    Write-Fail "Verification crashed: $_"
}

# ═══════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════
Write-Host ""
Write-Host "  ╔════════════════════════════════════╗" -ForegroundColor DarkCyan
Write-Host "  ║         Setup Complete!             ║" -ForegroundColor White
Write-Host "  ╚════════════════════════════════════╝" -ForegroundColor DarkCyan
Write-Host "  ✓ Passed:   $passed" -ForegroundColor Green
if ($warnings -gt 0) { Write-Host "  ⚠ Warnings: $warnings" -ForegroundColor Yellow }
if ($failed -gt 0)   { Write-Host "  ✗ Failed:   $failed" -ForegroundColor Red }
Write-Host ""

if ($failed -eq 0) {
    Write-Host "  🎉 Ready to go!" -ForegroundColor Green
    Write-Host ""
    Write-Host "  Launch the app:" -ForegroundColor White
    Write-Host "    → Double-click 'Voice Translator' on your Desktop" -ForegroundColor Cyan
    Write-Host "    → Or run: $APP_DIR\launch.bat" -ForegroundColor Cyan
    if ($script:GPU_NAME) {
        Write-Host ""
        Write-Host "  Your GPU: $($script:GPU_NAME) ($vramGB GB)" -ForegroundColor DarkGray
        Write-Host "  Model: Faster-Whisper $defaultModel ($computeType)" -ForegroundColor DarkGray
    }
} else {
    Write-Host "  ⚠  Some checks failed. Fix the issues above and run setup.bat again." -ForegroundColor Yellow
}
Write-Host ""

Pause-Continue
