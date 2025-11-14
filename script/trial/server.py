"""
Federated Learning Server for Pump Sensor Anomaly Detection
Uses FedAvg strategy to aggregate Convolutional Autoencoder models from clients
"""

import flwr as fl
from typing import Dict, Optional, Tuple, List
from flwr.common import Parameters, Scalar
import sys

def weighted_average(metrics: List[Tuple[int, Dict[str, Scalar]]]) -> Dict[str, Scalar]:
    """
    Aggregate evaluation metrics from all clients using weighted average.
    
    This function is called by the server after each evaluation round to combine
    the metrics (like loss and accuracy) reported by each client. The metrics are
    weighted by the number of samples each client used for evaluation.
    """
    # Extract losses and number of samples from each client
    losses = [num_samples * m["loss"] for num_samples, m in metrics]
    total_samples = sum([num_samples for num_samples, _ in metrics])
    
    # Calculate weighted average loss
    aggregated_loss = sum(losses) / total_samples if total_samples > 0 else 0
    
    print(f"  → Aggregated loss across all clients: {aggregated_loss:.4f}")
    
    return {"loss": aggregated_loss}


def main():
    """
    Start the Federated Learning server.
    
    The server coordinates the training process by:
    1. Waiting for clients to connect
    2. Sending the global model to clients for local training
    3. Receiving updated model weights from clients
    4. Aggregating the weights using FedAvg (weighted average)
    5. Repeating for multiple rounds
    """
    
    # Get number of training rounds from command line, default is 5
    num_rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    
    # Get minimum number of clients to wait for
    min_clients = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    
    print("="*70)
    print("FEDERATED LEARNING SERVER - PUMP SENSOR ANOMALY DETECTION")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  • Training rounds: {num_rounds}")
    print(f"  • Minimum clients required: {min_clients}")
    print(f"  • Strategy: FedAvg (Federated Averaging)")
    print(f"  • Server address: 0.0.0.0:8080")
    print(f"\nServer is starting and waiting for {min_clients} clients to connect...")
    print(f"Each client should run: python client.py <client_id> <server_ip>")
    print("="*70 + "\n")
    
    # Define the FedAvg strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,  # Use 100% of available clients for training
        fraction_evaluate=1.0,  # Use 100% of available clients for evaluation
        min_fit_clients=min_clients,  # Minimum clients needed to start training
        min_evaluate_clients=min_clients,  # Minimum clients needed for evaluation
        min_available_clients=min_clients,  # Wait for this many clients before starting
        evaluate_metrics_aggregation_fn=weighted_average,  # How to aggregate metrics
    )
    
    # Start the Flower server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\nThe federated learning process finished after {num_rounds} rounds.")
    print(f"Each client trained locally and the server aggregated the results.")
    print(f"The final global model represents collaborative learning without sharing raw data.")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

