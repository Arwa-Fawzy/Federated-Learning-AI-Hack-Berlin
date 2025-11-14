"""
Heterogeneous Data Generator for Federated Learning
Creates synthetic variations of pump sensor data to simulate multiple facilities/clients
with different operating conditions, sensor characteristics, and failure patterns.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import dirichlet
import os
import json
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class HeterogeneousDataGenerator:
    """
    Generate heterogeneous datasets for federated learning from a single source dataset.
    
    Strategies:
    1. Operating condition clustering
    2. Label distribution skew (Dirichlet)
    3. Sensor quality degradation
    4. Temporal/seasonal partitioning
    5. Feature subset heterogeneity
    6. Quantity skew
    """
    
    def __init__(self, data_path: str, seed: int = 42):
        """
        Initialize the generator.
        
        Args:
            data_path: Path to the original dataset (CSV)
            seed: Random seed for reproducibility
        """
        self.seed = seed
        np.random.seed(seed)
        
        # Load data
        print(f"Loading data from {data_path}...")
        self.df = pd.read_csv(data_path)
        self.original_size = len(self.df)
        
        # Get sensor columns (exclude timestamp, unnamed, machine_status)
        self.sensor_cols = [col for col in self.df.columns 
                           if col.startswith('sensor_') and col != 'sensor_15']  # sensor_15 is empty
        
        print(f"Dataset loaded: {len(self.df):,} rows, {len(self.sensor_cols)} sensors")
        print(f"Machine status distribution:\n{self.df['machine_status'].value_counts()}")
        
    def generate_clients(
        self,
        n_clients: int = 5,
        strategy: str = 'hybrid',
        alpha: float = 0.5,
        quantity_distribution: List[float] = None,
        output_dir: str = 'federated_data'
    ) -> Dict:
        """
        Generate heterogeneous client datasets.
        
        Args:
            n_clients: Number of clients/facilities to simulate
            strategy: Partitioning strategy - 'clustering', 'dirichlet', 'temporal', 'hybrid'
            alpha: Dirichlet concentration parameter (lower = more heterogeneous)
            quantity_distribution: Custom quantity distribution per client
            output_dir: Directory to save client datasets
            
        Returns:
            Dictionary with client statistics
        """
        print(f"\n{'='*80}")
        print(f"GENERATING {n_clients} HETEROGENEOUS CLIENTS - Strategy: {strategy.upper()}")
        print(f"{'='*80}\n")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Execute strategy
        if strategy == 'clustering':
            clients = self._strategy_clustering(n_clients)
        elif strategy == 'dirichlet':
            clients = self._strategy_dirichlet(n_clients, alpha)
        elif strategy == 'temporal':
            clients = self._strategy_temporal(n_clients)
        elif strategy == 'hybrid':
            clients = self._strategy_hybrid(n_clients, alpha, quantity_distribution)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Apply sensor degradation to some clients
        clients = self._apply_sensor_degradation(clients)
        
        # Apply feature subset heterogeneity
        clients = self._apply_feature_subsets(clients)
        
        # Save client datasets
        stats = self._save_clients(clients, output_dir)
        
        print(f"\n{'='*80}")
        print(f"CLIENT GENERATION COMPLETE")
        print(f"{'='*80}")
        print(f"Output directory: {output_dir}/")
        print(f"Total clients: {len(clients)}")
        
        return stats
    
    def _strategy_clustering(self, n_clients: int) -> Dict[int, pd.DataFrame]:
        """Partition by operating condition clustering"""
        print("Strategy: Operating Condition Clustering")
        print("- Clustering sensor data based on operating patterns\n")
        
        # Use a subset of key sensors for clustering (avoid NaN issues)
        clustering_sensors = [col for col in self.sensor_cols[:20] 
                             if self.df[col].notna().sum() > len(self.df) * 0.8]
        
        # Prepare data for clustering
        X = self.df[clustering_sensors].fillna(self.df[clustering_sensors].mean())
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # K-means clustering
        print(f"Running K-Means with {n_clients} clusters...")
        kmeans = KMeans(n_clusters=n_clients, random_state=self.seed, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Assign to clients
        clients = {}
        for i in range(n_clients):
            client_df = self.df[clusters == i].copy()
            clients[i] = client_df
            print(f"  Client {i}: {len(client_df):>7,} samples")
        
        return clients
    
    def _strategy_dirichlet(self, n_clients: int, alpha: float) -> Dict[int, pd.DataFrame]:
        """Partition using Dirichlet distribution for label skew"""
        print(f"Strategy: Dirichlet Label Distribution (alpha={alpha})")
        print("- Creating label imbalance across clients\n")
        
        # Get unique labels
        labels = self.df['machine_status'].values
        unique_labels = self.df['machine_status'].unique()
        n_classes = len(unique_labels)
        
        print(f"Classes: {unique_labels}")
        print(f"Alpha: {alpha} (lower = more heterogeneous)\n")
        
        # Sample from Dirichlet distribution
        label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)
        
        # Create client indices for each class
        clients = {i: [] for i in range(n_clients)}
        
        for class_idx, class_label in enumerate(unique_labels):
            # Get indices for this class
            class_indices = np.where(labels == class_label)[0]
            np.random.shuffle(class_indices)
            
            # Distribute according to Dirichlet proportions
            proportions = label_distribution[class_idx]
            proportions = proportions / proportions.sum()  # Normalize
            
            splits = np.cumsum(proportions * len(class_indices)).astype(int)[:-1]
            client_indices = np.split(class_indices, splits)
            
            for client_id, indices in enumerate(client_indices):
                clients[client_id].extend(indices)
        
        # Convert to DataFrames
        for client_id in clients:
            np.random.shuffle(clients[client_id])  # Shuffle to mix classes
            clients[client_id] = self.df.iloc[clients[client_id]].copy()
            print(f"  Client {client_id}: {len(clients[client_id]):>7,} samples")
        
        return clients
    
    def _strategy_temporal(self, n_clients: int) -> Dict[int, pd.DataFrame]:
        """Partition by temporal/seasonal periods"""
        print("Strategy: Temporal Partitioning")
        print("- Dividing data by time periods to simulate seasonal variations\n")
        
        # Convert timestamp to datetime
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self.df = self.df.sort_values('timestamp')
        
        # Split into equal time periods
        time_splits = np.array_split(self.df, n_clients)
        
        clients = {}
        for i, client_df in enumerate(time_splits):
            clients[i] = client_df.copy()
            start_date = client_df['timestamp'].min()
            end_date = client_df['timestamp'].max()
            print(f"  Client {i}: {len(client_df):>7,} samples ({start_date.date()} to {end_date.date()})")
        
        return clients
    
    def _strategy_hybrid(
        self, 
        n_clients: int, 
        alpha: float,
        quantity_distribution: List[float] = None
    ) -> Dict[int, pd.DataFrame]:
        """Hybrid approach: Clustering + Dirichlet + Quantity skew"""
        print(f"Strategy: HYBRID (Clustering + Dirichlet alpha={alpha} + Quantity Skew)")
        print("- Combining multiple heterogeneity sources for maximum realism\n")
        
        # Step 1: First do clustering (coarse partitioning)
        n_clusters = min(n_clients * 2, 10)  # More clusters than clients
        clustering_sensors = [col for col in self.sensor_cols[:20] 
                             if self.df[col].notna().sum() > len(self.df) * 0.8]
        
        X = self.df[clustering_sensors].fillna(self.df[clustering_sensors].mean())
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.seed, n_init=10)
        self.df['cluster'] = kmeans.fit_predict(X_scaled)
        
        # Step 2: Apply Dirichlet distribution within clusters
        clients = {i: [] for i in range(n_clients)}
        
        for cluster_id in range(n_clusters):
            cluster_data = self.df[self.df['cluster'] == cluster_id]
            if len(cluster_data) == 0:
                continue
            
            # Distribute this cluster to clients using Dirichlet
            cluster_indices = cluster_data.index.values
            
            # Sample proportions from Dirichlet
            proportions = np.random.dirichlet([alpha] * n_clients)
            proportions = proportions / proportions.sum()
            
            splits = np.cumsum(proportions * len(cluster_indices)).astype(int)[:-1]
            client_indices = np.split(cluster_indices, splits)
            
            for client_id, indices in enumerate(client_indices):
                clients[client_id].extend(indices)
        
        # Step 3: Apply quantity skew
        if quantity_distribution is None:
            # Default: Pareto-like distribution (realistic scenario)
            quantity_distribution = [0.05, 0.35, 0.15, 0.30, 0.15][:n_clients]
            quantity_distribution = quantity_distribution / np.sum(quantity_distribution)
        
        print("Applying quantity skew...")
        target_sizes = (np.array(quantity_distribution) * self.original_size).astype(int)
        
        # Resample clients to match target sizes
        for client_id in clients:
            current_indices = clients[client_id]
            target_size = target_sizes[client_id]
            
            if len(current_indices) < target_size:
                # Oversample
                additional = np.random.choice(current_indices, 
                                            size=target_size - len(current_indices),
                                            replace=True)
                current_indices = np.concatenate([current_indices, additional])
            else:
                # Undersample
                current_indices = np.random.choice(current_indices, 
                                                  size=target_size,
                                                  replace=False)
            
            clients[client_id] = current_indices
        
        # Convert to DataFrames
        for client_id in clients:
            np.random.shuffle(clients[client_id])
            clients[client_id] = self.df.iloc[clients[client_id]].copy()
            
            # Remove temporary cluster column
            if 'cluster' in clients[client_id].columns:
                clients[client_id] = clients[client_id].drop('cluster', axis=1)
            
            print(f"  Client {client_id}: {len(clients[client_id]):>7,} samples "
                  f"({len(clients[client_id])/self.original_size*100:.1f}% of total)")
        
        return clients
    
    def _apply_sensor_degradation(self, clients: Dict[int, pd.DataFrame]) -> Dict[int, pd.DataFrame]:
        """Apply sensor quality degradation to simulate different facility conditions"""
        print("\nApplying sensor degradation artifacts...")
        
        degradation_types = [
            ('Calibration Bias', 'bias'),
            ('Measurement Noise', 'noise'),
            ('Intermittent Dropout', 'dropout'),
            ('Signal Delay', 'delay'),
            ('None (Pristine)', 'none')
        ]
        
        for client_id, client_df in clients.items():
            deg_name, deg_type = degradation_types[client_id % len(degradation_types)]
            
            if deg_type == 'bias':
                # Add calibration bias to random sensors
                n_affected = np.random.randint(5, 15)
                affected_sensors = np.random.choice(self.sensor_cols, n_affected, replace=False)
                for sensor in affected_sensors:
                    if sensor in client_df.columns:
                        bias = np.random.uniform(-0.05, 0.10)  # ±5-10% bias
                        client_df[sensor] = client_df[sensor] * (1 + bias)
                print(f"  Client {client_id}: {deg_name} on {n_affected} sensors")
                
            elif deg_type == 'noise':
                # Add Gaussian noise
                n_affected = np.random.randint(5, 15)
                affected_sensors = np.random.choice(self.sensor_cols, n_affected, replace=False)
                for sensor in affected_sensors:
                    if sensor in client_df.columns and client_df[sensor].notna().any():
                        noise_std = client_df[sensor].std() * 0.10  # 10% of std
                        noise = np.random.normal(0, noise_std, len(client_df))
                        client_df[sensor] = client_df[sensor] + noise
                print(f"  Client {client_id}: {deg_name} on {n_affected} sensors")
                
            elif deg_type == 'dropout':
                # Random missing values
                n_affected = np.random.randint(3, 8)
                affected_sensors = np.random.choice(self.sensor_cols, n_affected, replace=False)
                dropout_rate = 0.15  # 15% missing
                for sensor in affected_sensors:
                    if sensor in client_df.columns:
                        mask = np.random.random(len(client_df)) < dropout_rate
                        client_df.loc[mask, sensor] = np.nan
                print(f"  Client {client_id}: {deg_name} on {n_affected} sensors ({dropout_rate*100:.0f}% missing)")
                
            elif deg_type == 'delay':
                # Time delay (shift values)
                n_affected = np.random.randint(3, 8)
                affected_sensors = np.random.choice(self.sensor_cols, n_affected, replace=False)
                delay_steps = np.random.randint(1, 5)
                for sensor in affected_sensors:
                    if sensor in client_df.columns:
                        client_df[sensor] = client_df[sensor].shift(delay_steps)
                print(f"  Client {client_id}: {deg_name} on {n_affected} sensors ({delay_steps} steps)")
                
            else:
                print(f"  Client {client_id}: {deg_name} (no degradation)")
            
            clients[client_id] = client_df
        
        return clients
    
    def _apply_feature_subsets(self, clients: Dict[int, pd.DataFrame]) -> Dict[int, pd.DataFrame]:
        """Simulate different sensor configurations across facilities"""
        print("\nApplying feature subset heterogeneity...")
        
        configurations = [
            ('Full Sensor Suite', 1.0),
            ('High-End Facility', 0.9),
            ('Standard Setup', 0.7),
            ('Budget Configuration', 0.5),
            ('Legacy System', 0.4)
        ]
        
        for client_id, client_df in clients.items():
            config_name, keep_ratio = configurations[client_id % len(configurations)]
            
            if keep_ratio < 1.0:
                # Randomly drop some sensors
                n_sensors = len(self.sensor_cols)
                n_keep = int(n_sensors * keep_ratio)
                kept_sensors = np.random.choice(self.sensor_cols, n_keep, replace=False)
                
                # Keep only selected sensors (plus essential columns)
                essential_cols = ['timestamp', 'machine_status']
                if 'Unnamed: 0' in client_df.columns:
                    essential_cols.append('Unnamed: 0')
                
                cols_to_keep = list(kept_sensors) + essential_cols
                cols_to_keep = [col for col in cols_to_keep if col in client_df.columns]
                
                client_df = client_df[cols_to_keep]
                print(f"  Client {client_id}: {config_name} ({n_keep}/{n_sensors} sensors)")
            else:
                print(f"  Client {client_id}: {config_name} (all sensors)")
            
            clients[client_id] = client_df
        
        return clients
    
    def _save_clients(self, clients: Dict[int, pd.DataFrame], output_dir: str) -> Dict:
        """Save client datasets and generate statistics"""
        print(f"\nSaving client datasets to {output_dir}/...")
        
        stats = {
            'n_clients': len(clients),
            'total_samples': 0,
            'clients': {}
        }
        
        for client_id, client_df in clients.items():
            # Save to CSV
            filename = f"client_{client_id}.csv"
            filepath = os.path.join(output_dir, filename)
            client_df.to_csv(filepath, index=False)
            
            # Collect statistics
            status_dist = client_df['machine_status'].value_counts().to_dict()
            n_sensors = len([col for col in client_df.columns if col.startswith('sensor_')])
            missing_rate = client_df.isnull().sum().sum() / (len(client_df) * len(client_df.columns))
            
            stats['clients'][client_id] = {
                'samples': len(client_df),
                'sensors': n_sensors,
                'status_distribution': status_dist,
                'missing_rate': float(missing_rate),
                'file': filename
            }
            stats['total_samples'] += len(client_df)
            
            print(f"  [OK] Client {client_id}: {filename} ({len(client_df):,} samples, "
                  f"{n_sensors} sensors)")
        
        # Save metadata
        metadata_path = os.path.join(output_dir, 'client_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"  [OK] Metadata: client_metadata.json")
        
        return stats


def main():
    """Example usage"""
    
    # Initialize generator
    generator = HeterogeneousDataGenerator(
        data_path='data/sensor.csv',
        seed=42
    )
    
    # Example 1: Hybrid strategy (recommended)
    print("\n" + "="*80)
    print("EXAMPLE 1: HYBRID STRATEGY")
    print("="*80)
    stats = generator.generate_clients(
        n_clients=5,
        strategy='hybrid',
        alpha=0.5,  # Moderate heterogeneity
        output_dir='federated_data/hybrid'
    )
    
    # Example 2: Pure clustering
    print("\n\n" + "="*80)
    print("EXAMPLE 2: CLUSTERING STRATEGY")
    print("="*80)
    stats = generator.generate_clients(
        n_clients=5,
        strategy='clustering',
        output_dir='federated_data/clustering'
    )
    
    # Example 3: Dirichlet with high heterogeneity
    print("\n\n" + "="*80)
    print("EXAMPLE 3: DIRICHLET STRATEGY (High Heterogeneity)")
    print("="*80)
    stats = generator.generate_clients(
        n_clients=5,
        strategy='dirichlet',
        alpha=0.1,  # Very heterogeneous
        output_dir='federated_data/dirichlet_high'
    )
    
    print("\n" + "="*80)
    print("ALL EXAMPLES COMPLETE!")
    print("="*80)
    print("\nGenerated datasets:")
    print("  - federated_data/hybrid/")
    print("  - federated_data/clustering/")
    print("  - federated_data/dirichlet_high/")


if __name__ == "__main__":
    main()

