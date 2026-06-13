# Backend'ni Windows service sifatida o'rnatadi (NSSM orqali).
# Administrator PowerShell'da bajariladi. 100% offline.
#
# Talab: NSSM (nssm.exe) shu papkada yoki PATH'da; venv tayyor (offline-wheels).

param(
    [string]$InstallDir = "C:\customs-ai",
    [string]$ServiceName = "CustomsAIBackend",
    [string]$Nssm = "nssm.exe"
)

$ErrorActionPreference = "Stop"

$Python = Join-Path $InstallDir "venv\Scripts\python.exe"
$AppDir = Join-Path $InstallDir "backend"

if (-not (Test-Path $Python)) {
    throw "venv topilmadi: $Python  (avval offline-wheels bilan o'rnating)"
}

Write-Host "Service o'rnatilmoqda: $ServiceName"
& $Nssm install $ServiceName $Python "-m" "uvicorn" "app.main:app" "--host" "127.0.0.1" "--port" "8000"
& $Nssm set $ServiceName AppDirectory $AppDir
& $Nssm set $ServiceName AppStdout (Join-Path $InstallDir "logs\backend.out.log")
& $Nssm set $ServiceName AppStderr (Join-Path $InstallDir "logs\backend.err.log")
& $Nssm set $ServiceName Start SERVICE_AUTO_START
# Faza 1 auth: faqat loopback (127.0.0.1) — tashqaridan kirib bo'lmaydi.

& $Nssm start $ServiceName
Write-Host "Tayyor. Health: http://127.0.0.1:8000/health"
