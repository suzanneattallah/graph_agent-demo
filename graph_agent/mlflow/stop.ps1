# Arrête et supprime le conteneur MLflow (les volumes sont conservés)
podman stop mlflow-server
podman rm mlflow-server
Write-Host "MLflow arrêté. Données conservées dans les volumes mlflow-db / mlflow-artifacts." -ForegroundColor Green
