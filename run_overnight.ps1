# ============================================================
#  run_overnight.ps1  Pipeline complet GraphAgent (nuit)
#  Lance depuis C:\Projet avec :  .\run_overnight.ps1
# ============================================================
#  Ordre d execution :
#    1. Nouvelles traces  (2_generer_traces.py)   agent 35b-a3b
#    2. GEPA Juges        (5_optimize_judges.py)  optimise les 4 prompts juges
#    3. Juges iter2       (3_juges_iteratif.py)   reevalue avec nouveaux prompts
#    4. Charts            (5_mlflow_charts.py)    courbes de convergence
#    5. Git push
# ============================================================

param(
    [switch]$SkipTraces,
    [switch]$SkipGepaJuges,
    [switch]$SkipJugesIter2,
    [switch]$SkipCharts
)

$ErrorActionPreference = "Continue"
$StartTime = Get-Date
$LogFile = "C:\Projet\overnight_run_$(Get-Date -Format 'yyyyMMdd_HHmm').log"

function Log {
    param([string]$msg, [string]$color = "White")
    $line = "[$(Get-Date -Format 'HH:mm:ss')] $msg"
    Write-Host $line -ForegroundColor $color
    Add-Content -Path $LogFile -Value $line
}

function RunStep {
    param([string]$label, [string]$module)
    Log "=== DEBUT : $label ===" "Cyan"
    $t = Get-Date
    python -m $module 2>&1 | ForEach-Object { Log "  $_" }
    $elapsed = [math]::Round(((Get-Date) - $t).TotalMinutes, 1)
    Log "=== FIN : $label ($elapsed min) ===" "Green"
}

# Anti-veille (SetThreadExecutionState via Python — ne nécessite pas admin)
Log "Configuration anti-veille..." "Yellow"
$kaProc = Start-Process python -ArgumentList "C:\Projet\keepawake.py" -WindowStyle Hidden -PassThru
Log "  keepawake.py lancé (PID $($kaProc.Id))" "Yellow"
# Fallback powercfg (peut nécessiter admin, erreur ignorée)
powercfg /change standby-timeout-ac 0 2>$null | Out-Null
powercfg /change monitor-timeout-ac 0 2>$null  | Out-Null

Log "Pipeline overnight demarre  log : $LogFile" "Yellow"
Log "Ordi : laisser allume, ecran peut etre ferme" "Yellow"

Set-Location C:\Projet

# Force UTF-8 pour Python (évite UnicodeEncodeError sur les outputs Windows cp1252)
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
if (-not $SkipTraces) {
    RunStep "Generation traces  qwen3.6-35b-a3b" "demo_client.2_generer_traces"
} else { Log "SKIP : Traces" "Gray" }

# 2  GEPA Juges
if (-not $SkipGepaJuges) {
    RunStep "GEPA  Optimisation prompts juges" "demo_client.5_optimize_judges"
} else { Log "SKIP : GEPA Juges" "Gray" }

# 3  Juges iteration 2 (auto-ok)
if (-not $SkipJugesIter2) {
    Log "=== DEBUT : Juges iteration 2 (auto-ok) ===" "Cyan"
    $t = Get-Date
    python -m demo_client.3_juges_iteratif --iterations 1 --limit 2 --auto-ok 2>&1 | ForEach-Object { Log "  $_" }
    $elapsed = [math]::Round(((Get-Date) - $t).TotalMinutes, 1)
    Log "=== FIN : Juges iteration 2 ($elapsed min) ===" "Green"
} else { Log "SKIP : Juges iter2" "Gray" }

# 4  Charts convergence
if (-not $SkipCharts) {
    RunStep "Charts MLflow � convergence" "demo_client.5_mlflow_charts"
} else { Log "SKIP : Charts" "Gray" }

# 5  Git push
Log "=== GIT PUSH ===" "Cyan"
Set-Location C:\Projet
git add -A
git commit -m "chore: overnight run $(Get-Date -Format 'yyyy-MM-dd')  traces 35b + GEPA juges + iter2 + charts"
git push origin master
Log "Git push termine" "Green"

$totalMin = [math]::Round(((Get-Date) - $StartTime).TotalMinutes, 1)
Log ""
Log "============================================" "Yellow"
Log "  PIPELINE TERMINE en $totalMin minutes"     "Yellow"
Log "  Log : $LogFile"                            "Yellow"
Log "  MLflow : http://localhost:5000"            "Yellow"
Log "============================================" "Yellow"
