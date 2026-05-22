# ============================================================
#  run_overnight.ps1  Pipeline complet GraphAgent (nuit)
#  Lance depuis C:\Projet avec :  .\run_overnight.ps1
# ============================================================
#  Ordre d execution :
#    1. Nouvelles traces   (2_generer_traces.py)   agent 35b-a3b
#    2. Juges iter1        (3_juges_iteratif.py)   baseline sur les nouvelles traces
#    3. GEPA Juges         (5_optimize_judges.py)  optimise les 4 prompts juges
#    4. Juges iter2        (3_juges_iteratif.py)   reevalue avec nouveaux prompts
#    5. Charts             (5_mlflow_charts.py)    courbes de convergence
#    6. Git push
# ============================================================

param(
    [switch]$SkipTraces,
    [switch]$SkipJugesIter1,
    [switch]$SkipGepaJuges,
    [switch]$SkipJugesIter2,
    [switch]$SkipCharts,
    [int]$JudgeLimit = 7
)

$ErrorActionPreference = "Stop"
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
    $exitCode = $LASTEXITCODE
    $elapsed = [math]::Round(((Get-Date) - $t).TotalMinutes, 1)
    if ($exitCode -ne 0) {
        throw "Étape '$label' en échec (exit code $exitCode)"
    }
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
    RunStep "Generation traces qwen3.6-35b-a3b" "demo_client.2_generer_traces"
} else { Log "SKIP : Traces" "Gray" }

# 2  Juges iteration 1 (baseline)
if (-not $SkipJugesIter1) {
    Log "=== DEBUT : Juges iteration 1 (auto-ok) ===" "Cyan"
    $t = Get-Date
    python -m demo_client.3_juges_iteratif --iterations 1 --limit $JudgeLimit --auto-ok 2>&1 | ForEach-Object { Log "  $_" }
    $exitCode = $LASTEXITCODE
    $elapsed = [math]::Round(((Get-Date) - $t).TotalMinutes, 1)
    if ($exitCode -ne 0) {
        throw "Étape 'Juges iteration 1' en échec (exit code $exitCode)"
    }
    Log "=== FIN : Juges iteration 1 ($elapsed min) ===" "Green"
} else { Log "SKIP : Juges iter1" "Gray" }

# 3  GEPA Juges
if (-not $SkipGepaJuges) {
    RunStep "GEPA Optimisation prompts juges" "demo_client.5_optimize_judges"
} else { Log "SKIP : GEPA Juges" "Gray" }

# 4  Juges iteration 2 (mêmes traces, prompts GEPA)
if (-not $SkipJugesIter2) {
    Log "=== DEBUT : Juges iteration 2 (auto-ok) ===" "Cyan"
    $t = Get-Date
    python -m demo_client.3_juges_iteratif --iterations 1 --limit $JudgeLimit --auto-ok 2>&1 | ForEach-Object { Log "  $_" }
    $exitCode = $LASTEXITCODE
    $elapsed = [math]::Round(((Get-Date) - $t).TotalMinutes, 1)
    if ($exitCode -ne 0) {
        throw "Étape 'Juges iteration 2' en échec (exit code $exitCode)"
    }
    Log "=== FIN : Juges iteration 2 ($elapsed min) ===" "Green"
} else { Log "SKIP : Juges iter2" "Gray" }

# 5  Charts convergence
if (-not $SkipCharts) {
    RunStep "Charts MLflow convergence" "demo_client.5_mlflow_charts"
} else { Log "SKIP : Charts" "Gray" }

# 6  Git push
Log "=== GIT PUSH ===" "Cyan"
Set-Location C:\Projet
git add -A
$pendingChanges = git status --porcelain
if ($pendingChanges) {
    git commit -m "chore: overnight run $(Get-Date -Format 'yyyy-MM-dd') traces + juges + GEPA + charts" -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
    if ($LASTEXITCODE -ne 0) {
        throw "git commit a échoué"
    }
    git push origin master
    if ($LASTEXITCODE -ne 0) {
        throw "git push a échoué"
    }
    Log "Git push termine" "Green"
} else {
    Log "Aucun changement a push" "Gray"
}

$totalMin = [math]::Round(((Get-Date) - $StartTime).TotalMinutes, 1)
Log ""
Log "============================================" "Yellow"
Log "  PIPELINE TERMINE en $totalMin minutes"     "Yellow"
Log "  Log : $LogFile"                            "Yellow"
Log "  MLflow : http://localhost:5000"            "Yellow"
Log "============================================" "Yellow"
