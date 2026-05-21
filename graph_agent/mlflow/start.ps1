# Build + run du conteneur MLflow (Podman)
# Exécution : .\mlflow\start.ps1  (depuis C:\Projet\graph_agent)

$ErrorActionPreference = "Stop"

$ImageName      = "mlflow-server:local"
$ContainerName  = "mlflow-server"
$DbVolume       = "mlflow-db"
$ArtifactsVolume= "mlflow-artifacts"

Write-Host "==> Build image $ImageName" -ForegroundColor Cyan
podman build -t $ImageName -f .\mlflow\Containerfile .\mlflow

Write-Host "==> Création des volumes" -ForegroundColor Cyan
podman volume create $DbVolume       | Out-Null
podman volume create $ArtifactsVolume| Out-Null

if (podman ps -a --format "{{.Names}}" | Select-String -Pattern "^$ContainerName$") {
    Write-Host "==> Conteneur existant détecté, suppression" -ForegroundColor Yellow
    podman rm -f $ContainerName | Out-Null
}

Write-Host "==> Lancement du conteneur" -ForegroundColor Cyan
podman run -d `
    --name $ContainerName `
    -p 5000:5000 `
    -v "${DbVolume}:/mlflow/db" `
    -v "${ArtifactsVolume}:/mlflow/artifacts" `
    $ImageName

Write-Host ""
Write-Host "MLflow UI : http://localhost:5000" -ForegroundColor Green
