Comando para iniciar el servidor de mlflow en background:
```bash
# Activar un entorno virtual en donde esté mlflow instalado
pyenv activate RedesNeuronales

# Correr con nohup y con > mlflow.log 2>&1 & para correrlo en background y no ver
# el log
nohup mlflow server --host 127.0.0.1 --port 8080 --backend-store-uri ./mlflow-storage/mlruns/ --artifacts-destination ./mlflow-storage/mlartifacts/ > mlflow.log 2>&1 &
```