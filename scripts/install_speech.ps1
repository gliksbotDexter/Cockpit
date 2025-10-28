<#
.SYNOPSIS
    Installs the speech-to-text (STT) and text-to-speech (TTS) runtime dependencies for Dexter.

.DESCRIPTION
    This script installs the packages required by `dexter_autonomy.speech` on Windows machines.
    It detects the active Python interpreter (preferring an activated virtual environment) and
    installs the following packages:

        - edge-tts (Edge neural voices)
        - pyttsx3 (Windows SAPI fallback)
        - SpeechRecognition (Windows STT wrapper)
        - pyaudio (microphone input for SpeechRecognition)
        - azure-cognitiveservices-speech (optional Azure voices/STT)
        - numpy (dependency for some SpeechRecognition backends)
        - sounddevice (optional mic helper)
        - openai-whisper (optional offline STT)

    Usage examples:
        # Install everything, including Azure and Whisper
        ./scripts/install_speech.ps1 -IncludeAzure -IncludeWhisper

        # Install only the free Windows/Edge stack
        ./scripts/install_speech.ps1

.PARAMETER Python
    Explicit path to the Python executable to use. Defaults to the interpreter in the current
    virtual environment, or the first `python`/`py` found on PATH.

.PARAMETER IncludeAzure
    When specified, installs the Azure Cognitive Services Speech SDK (`azure-cognitiveservices-speech`).

.PARAMETER IncludeWhisper
    When specified, installs OpenAI Whisper for offline STT (adds `openai-whisper`).

.PARAMETER Quiet
    Suppresses progress output from pip installs.

.NOTES
    - On Windows, installing PyAudio can fail without Microsoft Visual C++ build tools. This script
      will attempt to use `pipwin` to fetch a prebuilt wheel if necessary.
    - The script does not create or manage virtual environments; activate your venv first if desired.
#>
[CmdletBinding()]
param(
    [string]$Python,
    [switch]$IncludeAzure,
    [switch]$IncludeWhisper,
    [switch]$Quiet
)

function Get-PythonExe {
    param([string]$Explicit)
    if ([string]::IsNullOrWhiteSpace($Explicit) -eq $false) {
        return $Explicit
    }
    if ($env:VIRTUAL_ENV) {
        $candidate = Join-Path $env:VIRTUAL_ENV 'Scripts\python.exe'
        if (Test-Path $candidate) { return $candidate }
    }
    foreach ($cmd in @('python', 'py', 'python3')) {
        $which = (Get-Command $cmd -ErrorAction SilentlyContinue)
        if ($which) { return $which.Path }
    }
    throw "Unable to locate a Python interpreter. Activate your virtual environment or pass -Python explicitly."
}

function Invoke-PipInstall {
    param(
        [string]$PythonExe,
        [string[]]$Packages
    )
    if (-not $Packages) { return }
    $args = @('-m', 'pip', 'install', '--upgrade')
    if ($Quiet) { $args += '--quiet' }
    $args += $Packages
    Write-Host "[pip]" $PythonExe $args -ForegroundColor Cyan
    & $PythonExe $args
    if ($LASTEXITCODE -ne 0) {
        throw "pip install failed for packages: $($Packages -join ', ')"
    }
}

function Ensure-Pipwin {
    param([string]$PythonExe)
    $pipwinModule = & $PythonExe -m pip show pipwin 2>$null
    if (-not $pipwinModule) {
        Invoke-PipInstall -PythonExe $PythonExe -Packages @('pipwin')
    }
}

function Install-PyAudio {
    param([string]$PythonExe)
    Write-Host "Installing PyAudio (via pipwin)" -ForegroundColor Yellow
    Ensure-Pipwin -PythonExe $PythonExe
    & $PythonExe -m pipwin install pyaudio
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "pipwin could not install PyAudio; attempting pip install"
        Invoke-PipInstall -PythonExe $PythonExe -Packages @('pyaudio')
    }
}

try {
    $pythonExe = Get-PythonExe -Explicit $Python
    Write-Host "Using Python interpreter: $pythonExe" -ForegroundColor Green

    $corePackages = @(
        'edge-tts',
        'pyttsx3',
        'SpeechRecognition',
        'sounddevice',
        'numpy'
    )

    Invoke-PipInstall -PythonExe $pythonExe -Packages $corePackages

    Install-PyAudio -PythonExe $pythonExe

    if ($IncludeAzure) {
        Invoke-PipInstall -PythonExe $pythonExe -Packages @('azure-cognitiveservices-speech')
    }

    if ($IncludeWhisper) {
        Invoke-PipInstall -PythonExe $pythonExe -Packages @('openai-whisper')
    }

    Write-Host "Speech stack installation complete." -ForegroundColor Green
    Write-Host "If the Edge voice is not speaking, sign into Edge once and restart Dexter." -ForegroundColor Gray
}
catch {
    Write-Error $_
    exit 1
}
