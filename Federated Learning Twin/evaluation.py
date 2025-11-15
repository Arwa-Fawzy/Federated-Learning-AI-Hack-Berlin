# evaluation.py
"""
Evaluation and visualization for Federated Drift Twin demo.
Calls client_utils to load real data and client_app to simulate federated evaluation.
"""

import numpy as np
import matplotlib.pyplot as plt
from client_utils import load_and_prepare_sequences
from client_app import DriftTwinClient
import os

NUM_CLIENTS = 5
SEQ_LEN = 50
EPOCHS = 3
BATCH_SIZE = 32
NUM_ROUNDS = 6  # same as server

# --- Collect metrics from all clients per round ---
def simulate_federated_training():
    client_metrics = {cid: {"loss": [], "accuracy": [], "num_train": 0, "num_test": 0, "num_features": 0} for cid in range(NUM_CLIENTS)}

    for round_idx in range(NUM_ROUNDS):
        print(f"[evaluation] Simulating round {round_idx+1}/{NUM_ROUNDS}")
        for cid in range(NUM_CLIENTS):
            # Load client data
            x_train, x_hold, x_test, y_test, input_shape, scaler, meta = load_and_prepare_sequences(cid, seq_len=SEQ_LEN)
            
            # Store dataset info for table
            if round_idx == 0:  # only need once per client
                client_metrics[cid]["num_train"] = x_train.shape[0]
                client_metrics[cid]["num_test"] = x_test.shape[0]
                client_metrics[cid]["num_features"] = x_train.shape[2] if x_train.ndim == 3 else x_train.shape[1]

            # Initialize client
            client = DriftTwinClient(cid, epochs=EPOCHS, batch_size=BATCH_SIZE, seq_len=SEQ_LEN)
            client.x_train, client.x_hold, client.x_test, client.y_test = x_train, x_hold, x_test, y_test

            # Evaluate client
            loss, n, metrics = client.evaluate(client.get_parameters({}), {})
            loss = float(loss)  # ensure float
            client_metrics[cid]["loss"].append(loss)
            client_metrics[cid]["accuracy"].append(metrics["accuracy"])

    return client_metrics

# --- Aggregate metrics across clients ---
def aggregate_metrics(client_metrics):
    global_loss = []
    global_accuracy = []
    for r in range(NUM_ROUNDS):
        round_losses = [client_metrics[cid]["loss"][r] for cid in range(NUM_CLIENTS)]
        round_acc = [client_metrics[cid]["accuracy"][r] for cid in range(NUM_CLIENTS)]
        global_loss.append(np.mean(round_losses))
        global_accuracy.append(np.mean(round_acc))
    return np.array(global_loss), np.array(global_accuracy)

# --- Plot function ---
def plot_metrics(global_loss, global_accuracy):
    rounds = np.arange(1, NUM_ROUNDS+1)
    fig, ax1 = plt.subplots(figsize=(8,5))

    color = 'tab:blue'
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Global Loss (MSE)', color=color)
    ax1.plot(rounds, global_loss, marker='o', color=color, linewidth=2, label='Global Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks(rounds)
    ax1.grid(alpha=0.3)

    # Second y-axis for accuracy
    ax2 = ax1.twinx()
    color = 'tab:orange'
    ax2.set_ylabel('Global Accuracy', color=color)
    ax2.plot(rounds, global_accuracy, marker='s', color=color, linewidth=2, linestyle='--', label='Global Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)

    # Title
    plt.title("Federated Drift Twin: Global Loss & Accuracy per Round", fontsize=14, fontweight='bold')

    # Legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper right')

    # Annotate last points
    ax1.annotate(f"{global_loss[-1]:.6f}", xy=(rounds[-1], global_loss[-1]),
                 xytext=(5,5), textcoords='offset points', color='blue', fontweight='bold')
    ax2.annotate(f"{global_accuracy[-1]*100:.1f}%", xy=(rounds[-1], global_accuracy[-1]),
                 xytext=(5,-15), textcoords='offset points', color='orange', fontweight='bold')

    plt.tight_layout()
    plt.show()

# --- Display client dataset info as table ---
def print_client_table(client_metrics):
    print("\nClient Dataset Summary:")
    print(f"{'Client':>6} | {'#Train':>7} | {'#Test':>6} | {'#Features':>9} | {'Last Loss':>10} | {'Last Acc (%)':>12}")
    print("-"*60)
    for cid in range(NUM_CLIENTS):
        last_loss = client_metrics[cid]["loss"][-1]
        last_acc = client_metrics[cid]["accuracy"][-1]*100
        print(f"{cid:>6} | {client_metrics[cid]['num_train']:>7} | {client_metrics[cid]['num_test']:>6} | {client_metrics[cid]['num_features']:>9} | {last_loss:>10.6f} | {last_acc:>12.2f}")

# --- ENTRY POINT ---
if __name__ == "__main__":
    client_metrics = simulate_federated_training()
    global_loss, global_accuracy = aggregate_metrics(client_metrics)
    plot_metrics(global_loss, global_accuracy)
    print_client_table(client_metrics)
