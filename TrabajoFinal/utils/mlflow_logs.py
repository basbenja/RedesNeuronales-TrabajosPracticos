import mlflow
import tempfile

from pathlib import Path

def log_hyperparams(
    epochs, batch_size, optimizer, loss_fn, early_stopper=None, **extra_hyperparams
):
    hyperparams = {
        "max_epochs": epochs,
        "batch_size": batch_size,
        "optimizer": optimizer.__class__.__name__,
        "learning_rate": optimizer.defaults['lr'],
        "loss_function": loss_fn.__class__.__name__,
        "early_stopper": {
            "enabled": early_stopper is not None,
            "patience": early_stopper.patience if early_stopper else None,
            "delta": early_stopper.delta if early_stopper else None
        },
        **extra_hyperparams
    }
    mlflow.log_params(hyperparams)


def log_model_architecture(model):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp, "model_architecture.txt")
        path.write_text(str(model))
        mlflow.log_artifact(path)


def log_plot(fig, filename):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp, filename)
        fig.savefig(path)
        mlflow.log_artifact(path)


def log_losses(epoch, avg_loss_training, train_avg_loss, valid_avg_loss):
    mlflow.log_metric("Average Loss during Training", avg_loss_training, step=epoch)
    mlflow.log_metric("Average Loss in Training Set", train_avg_loss, step=epoch)
    mlflow.log_metric("Average Loss in Validation Set", valid_avg_loss, step=epoch)

def log_accs(epoch, acc_training, train_acc, valid_acc):
    mlflow.log_metric("Accuracy during Training", acc_training, step=epoch)
    mlflow.log_metric("Accuracy in Training Set", train_acc, step=epoch)
    mlflow.log_metric("Accuracy in Validation Set", valid_acc, step=epoch)