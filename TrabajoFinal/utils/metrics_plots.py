import matplotlib.pyplot as plt
import torch

from utils.mlflow_logs import log_plot

def losses_plot(avg_losses_training, train_avg_losses, valid_avg_losses, show=False):
    epochs = len(train_avg_losses)
    epochs_range = range(epochs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, avg_losses_training, label="Durante la época", linestyle="--", color="blue")
    plt.plot(epochs_range, train_avg_losses, label="Entrenamiento", linestyle="-", color="green")
    plt.plot(epochs_range, valid_avg_losses, label="Validación", linestyle=":", color="red")
    
    plt.xlabel("Épocas", fontdict={'fontsize': 16, 'fontweight': 'bold'})
    plt.ylabel("Pérdida promedio por lote", fontdict={'fontsize': 16, 'fontweight': 'bold'})
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=15)
    plt.grid()
    plt.tight_layout()

    fig = plt.gcf()
    log_plot(fig, "losses_plot.png")

    if show:
        plt.show()
    else:
        plt.close()

def accs_plot(accs_training, train_accs, valid_accs, show=False):
    epochs = len(train_accs)
    epochs_range = range(epochs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, accs_training, label="Durante la época", linestyle="--", color="blue")
    plt.plot(epochs_range, train_accs, label="Entrenamiento", linestyle="-", color="green")
    plt.plot(epochs_range, valid_accs, label="Validación", linestyle=":", color="red")
    
    plt.xlabel("Épocas", fontdict={'fontsize': 16, 'fontweight': 'bold'})
    plt.ylabel("Accuracy", fontdict={'fontsize': 16, 'fontweight': 'bold'})
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=15)
    plt.grid()
    plt.tight_layout()

    fig = plt.gcf()
    log_plot(fig, "accuracy_plot.png")

    if show:
        plt.show()
    else:
        plt.close()


def plot_orig_predicted(model, set, num_samples=3, show=False):
    model_device = next(model.parameters()).device
    model.eval()

    fig, axes = plt.subplots(nrows=num_samples, ncols=2, figsize=(8, 2*num_samples))

    for i in range(num_samples):
        sample_idx = torch.randint(len(set), size=(1,)).item()
        input = set[sample_idx][0].unsqueeze(1).to(model_device)
        
        with torch.no_grad():
            output = model(input).squeeze(1).cpu().numpy()

        # Plot original image
        ax_orig = axes[i, 0]
        ax_orig.imshow(input.squeeze().cpu().numpy(), cmap='gray')
        ax_orig.axis('off')
        ax_orig.set_title(f"Original {sample_idx}")
        
        # Plot predicted image
        ax_pred = axes[i, 1]
        ax_pred.imshow(output.squeeze(), cmap='gray')
        ax_pred.axis('off')
        ax_pred.set_title(f"Reconstructed {sample_idx}")

    log_plot(fig, "orig_predicted_plot.png")

    if show:
        plt.tight_layout()
        plt.show()
    else:
        plt.close()
