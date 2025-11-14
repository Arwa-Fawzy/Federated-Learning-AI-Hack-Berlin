"""
Federated Learning Client for Pump Sensor Anomaly Detection
Trains a Convolutional Autoencoder on local pump sensor data
"""

import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sys
import warnings
warnings.filterwarnings('ignore')


class ConvAutoencoder(nn.Module):
    """
    Convolutional Autoencoder for pump sensor data anomaly detection.
    
    The autoencoder learns to compress sensor readings into a lower-dimensional
    representation and then reconstruct them. Anomalies will have high
    reconstruction error since the model is trained primarily on normal data.
    
    Architecture:
    - Encoder: Conv1D layers that compress the sensor sequence
    - Bottleneck: Compressed latent representation
    - Decoder: Transposed Conv1D layers that reconstruct the original sequence
    """
    
    def __init__(self, num_sensors=20, sequence_length=10):
        super(ConvAutoencoder, self).__init__()
        
        # Encoder: Compresses the input sensor data
        self.encoder = nn.Sequential(
            nn.Conv1d(num_sensors, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),  # Reduces sequence length by half
            
            nn.Conv1d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),  # Further compression
        )
        
        # Decoder: Reconstructs the original sensor data
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(16, 32, kernel_size=2, stride=2),  # Upsampling
            nn.ReLU(),
            
            nn.ConvTranspose1d(32, num_sensors, kernel_size=2, stride=2),  # Restore original size
            nn.Sigmoid()  # Output between 0 and 1 (for normalized data)
        )
    
    def forward(self, x):
        """Forward pass through encoder and decoder"""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def load_and_prepare_data(client_id: int):
    """
    Load and prepare pump sensor data for the client.
    
    This function reads the client's specific dataset, selects sensor columns,
    handles missing values, normalizes the data, and creates sequences suitable
    for the convolutional autoencoder.
    
    Returns:
        train_loader: DataLoader for training data
        test_loader: DataLoader for testing data
        num_sensors: Number of sensor channels
        sequence_length: Length of each sensor sequence
    """
    
    print(f"\nLoading data for Client {client_id}...")
    
    # Load client-specific data
    df = pd.read_csv(f'../../federated_data/hybrid/client_{client_id}.csv')
    print(f"  • Dataset size: {len(df):,} samples")
    
    # Select sensor columns (excluding sensor_15 which is empty)
    sensor_cols = [col for col in df.columns if col.startswith('sensor_') and col != 'sensor_15']
    
    # Use a subset of sensors for faster training (20 sensors)
    sensor_cols = sensor_cols[:20]
    num_sensors = len(sensor_cols)
    
    print(f"  • Using {num_sensors} sensor channels")
    
    # Extract sensor data and handle missing values
    X = df[sensor_cols].fillna(df[sensor_cols].mean()).values
    
    # Normalize data to [0, 1] range (important for autoencoder)
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)
    X_normalized = (X_normalized - X_normalized.min()) / (X_normalized.max() - X_normalized.min())
    
    # Create sequences for convolutional processing
    sequence_length = 10  # Each sample is a sequence of 10 time steps
    sequences = []
    
    for i in range(len(X_normalized) - sequence_length + 1):
        seq = X_normalized[i:i+sequence_length]
        sequences.append(seq)
    
    sequences = np.array(sequences)
    print(f"  • Created {len(sequences):,} sequences of length {sequence_length}")
    
    # Split into train and test sets
    X_train, X_test = train_test_split(sequences, test_size=0.2, random_state=42)
    
    # Convert to PyTorch tensors (shape: [batch_size, num_sensors, sequence_length])
    X_train_tensor = torch.FloatTensor(X_train).permute(0, 2, 1)
    X_test_tensor = torch.FloatTensor(X_test).permute(0, 2, 1)
    
    # Create DataLoaders
    batch_size = 32
    train_dataset = TensorDataset(X_train_tensor, X_train_tensor)  # Autoencoder: input = output
    test_dataset = TensorDataset(X_test_tensor, X_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"  • Training batches: {len(train_loader)}")
    print(f"  • Testing batches: {len(test_loader)}")
    
    return train_loader, test_loader, num_sensors, sequence_length


def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train the autoencoder for one epoch.
    
    The model learns to reconstruct the input sensor data. Lower reconstruction
    error indicates the model has learned the normal patterns in the data.
    """
    model.train()
    total_loss = 0.0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss


def evaluate(model, test_loader, criterion, device):
    """
    Evaluate the autoencoder on test data.
    
    Returns the average reconstruction error on the test set.
    """
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(test_loader)
    return avg_loss


class PumpSensorClient(fl.client.NumPyClient):
    """
    Flower client that trains a Convolutional Autoencoder on pump sensor data.
    
    This client participates in federated learning by:
    1. Receiving global model parameters from the server
    2. Training the model on local data
    3. Sending updated parameters back to the server
    4. Evaluating the model and reporting metrics
    """
    
    def __init__(self, client_id, model, train_loader, test_loader, device):
        self.client_id = client_id
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.criterion = nn.MSELoss()  # Mean Squared Error for reconstruction
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    def get_parameters(self, config):
        """
        Return the current model parameters to the server.
        
        The server will aggregate these parameters with those from other clients
        using the FedAvg strategy.
        """
        return [val.cpu().numpy() for val in self.model.state_dict().values()]
    
    def set_parameters(self, parameters):
        """
        Update the local model with parameters received from the server.
        
        These are the aggregated parameters from all clients in the previous round.
        """
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config):
        """
        Train the model on local data.
        
        This function is called by the server in each training round.
        The client trains the model for a few epochs on its local dataset.
        """
        print(f"\n{'='*60}")
        print(f"Client {self.client_id}: Starting local training")
        print(f"{'='*60}")
        
        # Update model with global parameters
        self.set_parameters(parameters)
        
        # Train for multiple local epochs
        local_epochs = 3
        for epoch in range(local_epochs):
            train_loss = train_epoch(
                self.model, self.train_loader, 
                self.criterion, self.optimizer, self.device
            )
            print(f"  Epoch {epoch+1}/{local_epochs}: Training Loss = {train_loss:.4f}")
        
        print(f"  ✓ Local training complete for Client {self.client_id}")
        
        # Return updated model parameters
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}
    
    def evaluate(self, parameters, config):
        """
        Evaluate the model on local test data.
        
        This function is called by the server after each training round.
        The client evaluates the global model on its local test set.
        """
        print(f"\nClient {self.client_id}: Evaluating model...")
        
        # Update model with global parameters
        self.set_parameters(parameters)
        
        # Evaluate on test set
        test_loss = evaluate(self.model, self.test_loader, self.criterion, self.device)
        
        print(f"  → Test Loss (Reconstruction Error): {test_loss:.4f}")
        print(f"  ✓ Evaluation complete for Client {self.client_id}")
        
        # Return loss and number of test samples
        return float(test_loss), len(self.test_loader.dataset), {"loss": float(test_loss)}


def main():
    """
    Main function to start the federated learning client.
    
    Usage: python client.py <client_id> <server_ip>
    Example: python client.py 0 localhost
    """
    
    # Parse command line arguments
    if len(sys.argv) < 3:
        print("Usage: python client.py <client_id> <server_ip>")
        print("Example: python client.py 0 localhost")
        sys.exit(1)
    
    client_id = int(sys.argv[1])
    server_ip = sys.argv[2]
    
    print("="*70)
    print(f"FEDERATED LEARNING CLIENT {client_id} - PUMP SENSOR ANOMALY DETECTION")
    print("="*70)
    print(f"\nClient ID: {client_id}")
    print(f"Server Address: {server_ip}:8080")
    print(f"Model: Convolutional Autoencoder")
    print(f"Task: Anomaly Detection (Reconstruction-based)")
    
    # Set device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load and prepare data
    train_loader, test_loader, num_sensors, sequence_length = load_and_prepare_data(client_id)
    
    # Create model
    model = ConvAutoencoder(num_sensors=num_sensors, sequence_length=sequence_length).to(device)
    print(f"\nModel created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create Flower client
    client = PumpSensorClient(client_id, model, train_loader, test_loader, device)
    
    print(f"\n{'='*70}")
    print(f"Client {client_id} ready. Connecting to server at {server_ip}:8080...")
    print(f"{'='*70}\n")
    
    # Start Flower client (this will block until training is complete)
    fl.client.start_numpy_client(
        server_address=f"{server_ip}:8080",
        client=client
    )
    
    print(f"\n{'='*70}")
    print(f"Client {client_id}: Federated learning complete!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

