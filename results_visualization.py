import re
import matplotlib.pyplot as plt

def parse_results(file_path):
    epochs = []
    train_losses = []
    val_losses = []
    aucs = []
    auprs = []

    with open(file_path, 'r') as file:
        for line in file:
            if 'Training_avg' in line:
                # Extract epoch
                epoch_match = re.search(r'epoch: (\d+)', line)
                if epoch_match:
                    epochs.append(int(epoch_match.group(1)))

                # Extract training loss
                train_loss_match = re.search(r'label_loss: ([\d\.]+)', line)
                if train_loss_match:
                    loss = float(train_loss_match.group(1))
                    train_losses.append(min(loss, 1))

                # Extract validation loss
                val_loss_match = re.search(r'Validation_avg: loss: ([\d\.]+)', line)
                if val_loss_match:
                    loss = float(val_loss_match.group(1))
                    val_losses.append(min(loss, 1))

                # Extract AUC
                auc_match = re.search(r'AUC: ([\d\.]+)', line)
                if auc_match:
                    aucs.append(float(auc_match.group(1)))

                # Extract AUPR
                aupr_match = re.search(r'AUPR: \[([\d\.]+)\]', line)
                if aupr_match:
                    auprs.append(float(aupr_match.group(1)))

    return epochs, train_losses, val_losses, aucs, auprs


def plot_metrics(epochs, train_losses, val_losses, aucs, auprs, graph_type):
    plt.figure(figsize=(12, 10))

    # Plot training and validation loss
    # plt.subplot(2, 2, 1)
    # plt.plot(epochs, train_losses, label='Training Loss', color='blue')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Training Loss Over Epochs')
    # plt.legend()

    # plt.subplot(2, 2, 2)
    # plt.plot(epochs, val_losses, label='Validation Loss', color='orange')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Validation Loss Over Epochs')
    # plt.legend()

    # Plot training and validation loss together
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss', color='blue')
    plt.plot(epochs, val_losses, label='Validation Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{graph_type} Training and Validation Loss Over Epochs')
    plt.legend()

    # Plot AUC
    plt.subplot(2, 2, 3)
    plt.plot(epochs, aucs, label='AUC', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.title(f'{graph_type} AUC Over Epochs')
    plt.legend()

    # Plot AUPR
    plt.subplot(2, 2, 4)
    plt.plot(epochs, auprs, label='AUPR', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('AUPR')
    plt.title(f'{graph_type} AUPR Over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'{graph_type}_metrics.png')
    plt.show()


if __name__ == '__main__':
    file_path = './khop_structure_backbone_result/gnn_training_seed_1/valid_results.txt'
    epochs, train_losses, val_losses, aucs, auprs = parse_results(file_path)
    plot_metrics(epochs, train_losses, val_losses, aucs, auprs, "Sequence + 1-hop-structure")
