"""
Example Federated Learning Implementation
Using the generated heterogeneous pump sensor data

This is a simple example showing how to integrate the heterogeneous data
with a federated learning workflow.
"""

import pandas as pd
import numpy as np
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
import warnings
warnings.filterwarnings('ignore')


class FederatedLearningSimulator:
    """
    Simple federated learning simulator for pump sensor data
    Uses RandomForest as base model (can be replaced with neural networks)
    """
    
    def __init__(self, data_dir: str = 'federated_data/hybrid'):
        """
        Initialize FL simulator.
        
        Args:
            data_dir: Directory containing client datasets
        """
        self.data_dir = data_dir
        self.clients = {}
        self.global_model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Load clients
        self._load_clients()
        
        # Prepare data
        self._prepare_data()
    
    def _load_clients(self):
        """Load all client datasets"""
        print("Loading client datasets...")
        
        for filename in os.listdir(self.data_dir):
            if filename.startswith('client_') and filename.endswith('.csv'):
                client_id = int(filename.split('_')[1].split('.')[0])
                filepath = os.path.join(self.data_dir, filename)
                
                df = pd.read_csv(filepath)
                print(f"  Client {client_id}: {len(df):,} samples")
                
                self.clients[client_id] = {
                    'data': df,
                    'model': None,
                    'accuracy': 0.0
                }
    
    def _prepare_data(self):
        """Prepare data for training"""
        print("\nPreparing data...")
        
        # Get all sensor columns from first client
        first_client = self.clients[0]['data']
        sensor_cols = [col for col in first_client.columns if col.startswith('sensor_')]
        
        # Find common sensors across all clients
        for client_id in self.clients:
            client_sensors = [col for col in self.clients[client_id]['data'].columns 
                            if col.startswith('sensor_')]
            sensor_cols = list(set(sensor_cols) & set(client_sensors))
        
        print(f"  Common sensors across all clients: {len(sensor_cols)}")
        
        # Prepare each client's data
        for client_id in self.clients:
            df = self.clients[client_id]['data']
            
            # Select features (common sensors only)
            X = df[sensor_cols].fillna(df[sensor_cols].mean())
            
            # Encode labels
            y = df['machine_status']
            
            # Split into train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            self.clients[client_id]['X_train'] = X_train
            self.clients[client_id]['X_test'] = X_test
            self.clients[client_id]['y_train'] = y_train
            self.clients[client_id]['y_test'] = y_test
            self.clients[client_id]['features'] = sensor_cols
            
            print(f"  Client {client_id}: Train={len(X_train):,}, Test={len(X_test):,}")
    
    def train_centralized(self):
        """
        Baseline: Centralized training (all data pooled)
        """
        print("\n" + "="*80)
        print("CENTRALIZED TRAINING (Baseline)")
        print("="*80)
        
        # Pool all training data
        X_train_all = []
        y_train_all = []
        X_test_all = []
        y_test_all = []
        
        for client_id in self.clients:
            X_train_all.append(self.clients[client_id]['X_train'])
            y_train_all.append(self.clients[client_id]['y_train'])
            X_test_all.append(self.clients[client_id]['X_test'])
            y_test_all.append(self.clients[client_id]['y_test'])
        
        X_train = pd.concat(X_train_all, ignore_index=True)
        y_train = pd.concat(y_train_all, ignore_index=True)
        X_test = pd.concat(X_test_all, ignore_index=True)
        y_test = pd.concat(y_test_all, ignore_index=True)
        
        print(f"\nTotal training samples: {len(X_train):,}")
        print(f"Total test samples: {len(X_test):,}")
        
        # Standardize
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Encode labels
        self.label_encoder.fit(y_train)
        y_train_encoded = self.label_encoder.transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        # Train model
        print("\nTraining centralized model...")
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train_scaled, y_train_encoded)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test_encoded, y_pred)
        f1 = f1_score(y_test_encoded, y_pred, average='weighted')
        
        print(f"\n✓ Centralized Model Performance:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test_encoded, y_pred, 
                                   target_names=self.label_encoder.classes_))
        
        return accuracy, f1
    
    def train_local_only(self):
        """
        Local-only training (each client trains independently)
        """
        print("\n" + "="*80)
        print("LOCAL-ONLY TRAINING")
        print("="*80)
        
        results = []
        
        for client_id in sorted(self.clients.keys()):
            print(f"\nClient {client_id}:")
            
            X_train = self.clients[client_id]['X_train']
            X_test = self.clients[client_id]['X_test']
            y_train = self.clients[client_id]['y_train']
            y_test = self.clients[client_id]['y_test']
            
            # Standardize locally
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Encode labels
            label_encoder = LabelEncoder()
            y_train_encoded = label_encoder.fit_transform(y_train)
            y_test_encoded = label_encoder.transform(y_test)
            
            # Train local model
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X_train_scaled, y_train_encoded)
            
            # Evaluate
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test_encoded, y_pred)
            f1 = f1_score(y_test_encoded, y_pred, average='weighted')
            
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  F1 Score: {f1:.4f}")
            
            results.append({
                'client_id': client_id,
                'accuracy': accuracy,
                'f1_score': f1,
                'train_size': len(X_train),
                'test_size': len(X_test)
            })
            
            self.clients[client_id]['model'] = model
            self.clients[client_id]['accuracy'] = accuracy
        
        # Average performance
        avg_accuracy = np.mean([r['accuracy'] for r in results])
        avg_f1 = np.mean([r['f1_score'] for r in results])
        
        print(f"\n✓ Local-Only Average Performance:")
        print(f"  Avg Accuracy: {avg_accuracy:.4f}")
        print(f"  Avg F1 Score: {avg_f1:.4f}")
        
        return results
    
    def train_federated_simple(self):
        """
        Simplified federated learning:
        1. Each client trains locally
        2. Ensemble predictions (simple aggregation)
        """
        print("\n" + "="*80)
        print("FEDERATED LEARNING (Simple Ensemble)")
        print("="*80)
        
        # First do local training
        print("\nStep 1: Local training on each client...")
        self.train_local_only()
        
        # Step 2: Federated evaluation (ensemble predictions)
        print("\nStep 2: Federated evaluation (ensemble)...")
        
        results = []
        
        for client_id in sorted(self.clients.keys()):
            X_test = self.clients[client_id]['X_test']
            y_test = self.clients[client_id]['y_test']
            
            # Collect predictions from all clients
            predictions = []
            
            for other_client_id in self.clients:
                if self.clients[other_client_id]['model'] is None:
                    continue
                
                # Get other client's model
                other_model = self.clients[other_client_id]['model']
                
                # Standardize with other client's scaler (simulated)
                scaler = StandardScaler()
                scaler.fit(self.clients[other_client_id]['X_train'])
                X_test_scaled = scaler.transform(X_test)
                
                # Predict
                pred = other_model.predict(X_test_scaled)
                predictions.append(pred)
            
            # Ensemble: Majority voting
            predictions = np.array(predictions)
            ensemble_pred = np.apply_along_axis(
                lambda x: np.bincount(x).argmax(),
                axis=0,
                arr=predictions
            )
            
            # Evaluate
            label_encoder = LabelEncoder()
            y_test_encoded = label_encoder.fit_transform(y_test)
            
            accuracy = accuracy_score(y_test_encoded, ensemble_pred)
            f1 = f1_score(y_test_encoded, ensemble_pred, average='weighted')
            
            results.append({
                'client_id': client_id,
                'accuracy': accuracy,
                'f1_score': f1
            })
            
            print(f"  Client {client_id} (Federated): Accuracy={accuracy:.4f}, F1={f1:.4f}")
        
        # Average performance
        avg_accuracy = np.mean([r['accuracy'] for r in results])
        avg_f1 = np.mean([r['f1_score'] for r in results])
        
        print(f"\n✓ Federated Ensemble Performance:")
        print(f"  Avg Accuracy: {avg_accuracy:.4f}")
        print(f"  Avg F1 Score: {avg_f1:.4f}")
        
        return results
    
    def compare_approaches(self):
        """
        Compare all three approaches
        """
        print("\n" + "="*80)
        print("COMPARING APPROACHES")
        print("="*80)
        
        # Centralized
        cent_acc, cent_f1 = self.train_centralized()
        
        # Local-only
        local_results = self.train_local_only()
        local_avg_acc = np.mean([r['accuracy'] for r in local_results])
        local_avg_f1 = np.mean([r['f1_score'] for r in local_results])
        
        # Federated
        fed_results = self.train_federated_simple()
        fed_avg_acc = np.mean([r['accuracy'] for r in fed_results])
        fed_avg_f1 = np.mean([r['f_score'] for r in fed_results])
        
        # Summary
        print("\n" + "="*80)
        print("FINAL COMPARISON")
        print("="*80)
        
        print(f"\n{'Approach':<20} {'Accuracy':<12} {'F1 Score':<12}")
        print("-" * 44)
        print(f"{'Centralized':<20} {cent_acc:.4f}      {cent_f1:.4f}")
        print(f"{'Local-Only':<20} {local_avg_acc:.4f}      {local_avg_f1:.4f}")
        print(f"{'Federated':<20} {fed_avg_acc:.4f}      {fed_avg_f1:.4f}")
        
        print("\n" + "="*80)
        print("INSIGHTS")
        print("="*80)
        
        print("\n✓ Centralized: Best performance (all data pooled)")
        print("✓ Federated: Competitive performance without data sharing")
        print("✓ Local-Only: Lower performance (limited data per client)")
        
        print("\n💡 Federated Learning Benefits:")
        print("  - Privacy preservation (no raw data sharing)")
        print("  - Leverages distributed knowledge")
        print("  - Scalable to many clients")
        print("  - Reduces communication overhead")


def main():
    """Example usage"""
    
    data_dir = 'federated_data/hybrid'
    
    if not os.path.exists(data_dir):
        print(f"Error: {data_dir} not found.")
        print("Please run: python heterogeneous_data_generator.py")
        return
    
    print("="*80)
    print("FEDERATED LEARNING ON HETEROGENEOUS PUMP SENSOR DATA")
    print("="*80)
    
    # Initialize simulator
    fl_sim = FederatedLearningSimulator(data_dir=data_dir)
    
    # Run comparisons
    fl_sim.compare_approaches()
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)
    
    print("\n📝 Notes:")
    print("  - This is a simplified FL simulation for demonstration")
    print("  - For production, use Flower, TensorFlow Federated, or PySyft")
    print("  - Neural networks (LSTM, CNN) may perform better for time-series")
    print("  - Consider FedProx, FedAdam for better convergence on non-IID data")
    
    print("\n🚀 Next Steps:")
    print("  1. Implement proper FedAvg with weight aggregation")
    print("  2. Add differential privacy for secure aggregation")
    print("  3. Experiment with different models (LSTM, CNN)")
    print("  4. Test various FL algorithms (FedProx, FedAdam)")
    print("  5. Analyze convergence on heterogeneous data")


if __name__ == "__main__":
    main()

