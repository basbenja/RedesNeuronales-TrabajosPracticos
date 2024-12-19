import mlflow
import torch
import torch.nn as nn

from tqdm.notebook import tqdm
from utils.mlflow_logs import log_losses, log_accs

def train_step(model, dataloader, loss_fn, optimizer):
    """
    Trains the model for ONE epoch = goes over the dataset one time using the batches.
    """
    num_batches = len(dataloader)
    size = len(dataloader.dataset)
    
    is_classification = isinstance(loss_fn, nn.CrossEntropyLoss)
    total_acc = 0 if is_classification else None
    total_loss = 0
    
    model_device = next(model.parameters()).device
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(model_device), y.to(model_device)

        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        if is_classification:
            total_acc += (y_pred.argmax(1) == y).float().sum().item()

        if batch % 100 == 0:
            print(f"Batch {batch} of {num_batches}. Loss in batch: {loss.item():.4f}")
            if is_classification:
                print(f"Accuracy in batch: {(y_pred.argmax(1) == y).float().mean().item():.4f}")

    avg_loss = total_loss / num_batches
    acc = total_acc / size if is_classification else None

    return acc, avg_loss


def validate_step(model, dataloader, loss_fn):
    num_batches = len(dataloader)
    size = len(dataloader.dataset)
    
    is_classification = isinstance(loss_fn, nn.CrossEntropyLoss)
    total_acc = 0 if is_classification else None
    total_loss = 0

    model_device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(model_device), y.to(model_device)

            y_pred = model(X)
            total_loss += loss_fn(y_pred, y).item()

            if is_classification:
                total_acc += (y_pred.argmax(1) == y).float().sum().item()

    avg_loss = total_loss / num_batches
    acc = total_acc / size if is_classification else None

    return acc, avg_loss


def train_validate_loop(
    model, train_loader, valid_loader, loss_fn, optimizer, epochs, early_stopper=None,
):
    is_classification = isinstance(loss_fn, nn.CrossEntropyLoss)

    if is_classification:
        accs_training, train_accs, valid_accs = [], [], []
    else:
        accs_training = train_accs = valid_accs = None

    avg_losses_training, train_avg_losses, valid_avg_losses = ([], [], [])

    early_stopped = False
    for epoch in tqdm(range(epochs)):
        tqdm.write(f"Epoch {epoch}")
        acc_training, avg_loss_training = train_step(model, train_loader, loss_fn, optimizer)

        train_acc, train_avg_loss = validate_step(model, train_loader, loss_fn)
        valid_acc, valid_avg_loss = validate_step(model, valid_loader, loss_fn)

        avg_losses_training.append(avg_loss_training)
        train_avg_losses.append(train_avg_loss)
        valid_avg_losses.append(valid_avg_loss)
        log_losses(epoch, avg_loss_training, train_avg_loss, valid_avg_loss)
        tqdm.write(f"Train avg loss: {train_avg_loss:.6f} | Valid avg loss: {valid_avg_loss:.6f}")
        
        if is_classification:
            accs_training.append(acc_training)
            train_accs.append(train_acc)
            valid_accs.append(valid_acc)
            log_accs(epoch, acc_training, train_acc, valid_acc)
            tqdm.write(f"Train accuracy: {train_acc:.6f} | Valid accuracy: {valid_acc:.6f}")
 
        if early_stopper and early_stopper.update(model, epoch, valid_avg_loss):
            early_stopped = True
            model.load_state_dict(early_stopper.best_model)
            tqdm.write(
                "Early stopping activated. \n"
                f"Best epoch: {early_stopper.best_epoch}\n"
                f"Best loss: {early_stopper.best_loss:.6f}\n"
            )
            break

        tqdm.write("----------------------------------------------------------------")

    mlflow.log_params({
        "last_epoch": epoch,
        "early_stopped": early_stopped,
    })
    print(f"Training finished! Trained for {epoch} epochs.")
    print(
        f"Final results from epoch {epoch}:\n"
        f"  - Train avg loss: {train_avg_loss:.6f}\n"
        f"  - Valid avg loss: {valid_avg_loss:.6f}"
    )

    return {
        'model': model,
        'avg_losses_training': avg_losses_training,
        'train_avg_losses': train_avg_losses,
        'valid_avg_losses': valid_avg_losses,
        'accs_training': accs_training,
        'train_accs': train_accs,
        'valid_accs': valid_accs
    }