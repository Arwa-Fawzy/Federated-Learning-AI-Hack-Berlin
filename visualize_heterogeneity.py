"""
Visualization tools for heterogeneous federated learning data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from scipy.stats import wasserstein_distance, ks_2samp
from typing import Dict, List


class HeterogeneityVisualizer:
    """Visualize and analyze heterogeneity across federated clients"""
    
    def __init__(self, data_dir: str):
        """
        Initialize visualizer with client data directory.
        
        Args:
            data_dir: Directory containing client CSV files and metadata
        """
        self.data_dir = data_dir
        self.clients = {}
        self.metadata = None
        
        # Load metadata
        metadata_path = os.path.join(data_dir, 'client_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        
        # Load client data
        self._load_clients()
        
    def _load_clients(self):
        """Load all client datasets"""
        print(f"Loading clients from {self.data_dir}...")
        for filename in os.listdir(self.data_dir):
            if filename.startswith('client_') and filename.endswith('.csv'):
                client_id = int(filename.split('_')[1].split('.')[0])
                filepath = os.path.join(self.data_dir, filename)
                self.clients[client_id] = pd.read_csv(filepath)
                print(f"  Loaded Client {client_id}: {len(self.clients[client_id]):,} samples")
    
    def plot_data_distribution(self, save_path: str = None):
        """Plot data distribution across clients"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Data Distribution Across Clients', fontsize=16, fontweight='bold')
        
        # 1. Sample counts
        ax = axes[0, 0]
        client_ids = sorted(self.clients.keys())
        counts = [len(self.clients[cid]) for cid in client_ids]
        colors = plt.cm.Set3(np.linspace(0, 1, len(client_ids)))
        
        bars = ax.bar(client_ids, counts, color=colors, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Client ID', fontsize=12)
        ax.set_ylabel('Number of Samples', fontsize=12)
        ax.set_title('Sample Count per Client', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height):,}',
                   ha='center', va='bottom', fontsize=9)
        
        # 2. Label distribution
        ax = axes[0, 1]
        label_data = []
        for cid in client_ids:
            status_counts = self.clients[cid]['machine_status'].value_counts()
            label_data.append([
                status_counts.get('NORMAL', 0),
                status_counts.get('RECOVERING', 0),
                status_counts.get('BROKEN', 0)
            ])
        
        label_data = np.array(label_data)
        label_percentages = label_data / label_data.sum(axis=1, keepdims=True) * 100
        
        x = np.arange(len(client_ids))
        width = 0.6
        
        bottom = np.zeros(len(client_ids))
        colors_status = ['#2ecc71', '#f39c12', '#e74c3c']
        labels_status = ['NORMAL', 'RECOVERING', 'BROKEN']
        
        for i, (color, label) in enumerate(zip(colors_status, labels_status)):
            ax.bar(x, label_percentages[:, i], width, bottom=bottom,
                  label=label, color=color, edgecolor='black', alpha=0.7)
            bottom += label_percentages[:, i]
        
        ax.set_xlabel('Client ID', fontsize=12)
        ax.set_ylabel('Percentage (%)', fontsize=12)
        ax.set_title('Label Distribution (% per Client)', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(client_ids)
        ax.legend(loc='upper right')
        ax.grid(axis='y', alpha=0.3)
        
        # 3. Sensor availability
        ax = axes[1, 0]
        sensor_counts = []
        for cid in client_ids:
            n_sensors = len([col for col in self.clients[cid].columns 
                           if col.startswith('sensor_')])
            sensor_counts.append(n_sensors)
        
        bars = ax.bar(client_ids, sensor_counts, color=colors, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Client ID', fontsize=12)
        ax.set_ylabel('Number of Sensors', fontsize=12)
        ax.set_title('Sensor Availability per Client', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=9)
        
        # 4. Missing data rate
        ax = axes[1, 1]
        missing_rates = []
        for cid in client_ids:
            missing_rate = self.clients[cid].isnull().sum().sum() / \
                          (len(self.clients[cid]) * len(self.clients[cid].columns)) * 100
            missing_rates.append(missing_rate)
        
        bars = ax.bar(client_ids, missing_rates, color=colors, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Client ID', fontsize=12)
        ax.set_ylabel('Missing Data Rate (%)', fontsize=12)
        ax.set_title('Data Quality: Missing Values per Client', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}%',
                   ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.show()
    
    def plot_feature_distributions(self, sensor_name: str = 'sensor_00', save_path: str = None):
        """Plot feature distribution comparison across clients"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'Feature Distribution Comparison: {sensor_name}', 
                    fontsize=16, fontweight='bold')
        
        client_ids = sorted(self.clients.keys())
        colors = plt.cm.Set3(np.linspace(0, 1, len(client_ids)))
        
        # Collect data
        data_by_client = {}
        for cid in client_ids:
            if sensor_name in self.clients[cid].columns:
                data_by_client[cid] = self.clients[cid][sensor_name].dropna()
        
        if not data_by_client:
            print(f"Warning: {sensor_name} not found in any client")
            return
        
        # 1. Overlapping histograms
        ax = axes[0, 0]
        for cid, color in zip(client_ids, colors):
            if cid in data_by_client:
                ax.hist(data_by_client[cid], bins=50, alpha=0.5, 
                       label=f'Client {cid}', color=color, edgecolor='black')
        ax.set_xlabel(sensor_name, fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Histogram Overlay', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 2. KDE plot
        ax = axes[0, 1]
        for cid, color in zip(client_ids, colors):
            if cid in data_by_client and len(data_by_client[cid]) > 1:
                data_by_client[cid].plot.kde(ax=ax, label=f'Client {cid}', 
                                            color=color, linewidth=2)
        ax.set_xlabel(sensor_name, fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title('Kernel Density Estimate', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 3. Box plot
        ax = axes[0, 2]
        data_list = [data_by_client[cid] for cid in client_ids if cid in data_by_client]
        bp = ax.boxplot(data_list, labels=[f'C{cid}' for cid in client_ids if cid in data_by_client],
                       patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_xlabel('Client ID', fontsize=12)
        ax.set_ylabel(sensor_name, fontsize=12)
        ax.set_title('Box Plot Comparison', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # 4. Violin plot
        ax = axes[1, 0]
        positions = list(range(len([cid for cid in client_ids if cid in data_by_client])))
        parts = ax.violinplot(data_list, positions=positions, showmeans=True, showmedians=True)
        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        ax.set_xticks(positions)
        ax.set_xticklabels([f'C{cid}' for cid in client_ids if cid in data_by_client])
        ax.set_xlabel('Client ID', fontsize=12)
        ax.set_ylabel(sensor_name, fontsize=12)
        ax.set_title('Violin Plot', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # 5. Statistical summary
        ax = axes[1, 1]
        stats_data = []
        for cid in client_ids:
            if cid in data_by_client:
                stats_data.append([
                    data_by_client[cid].mean(),
                    data_by_client[cid].std(),
                    data_by_client[cid].min(),
                    data_by_client[cid].max()
                ])
        
        stats_df = pd.DataFrame(stats_data, 
                               columns=['Mean', 'Std', 'Min', 'Max'],
                               index=[f'C{cid}' for cid in client_ids if cid in data_by_client])
        
        x = np.arange(len(stats_df))
        width = 0.2
        
        ax.bar(x - 1.5*width, stats_df['Mean'], width, label='Mean', alpha=0.7)
        ax.bar(x - 0.5*width, stats_df['Std'], width, label='Std', alpha=0.7)
        ax.bar(x + 0.5*width, stats_df['Min'], width, label='Min', alpha=0.7)
        ax.bar(x + 1.5*width, stats_df['Max'], width, label='Max', alpha=0.7)
        
        ax.set_xlabel('Client ID', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.set_title('Statistical Summary', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(stats_df.index)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # 6. Cumulative distribution
        ax = axes[1, 2]
        for cid, color in zip(client_ids, colors):
            if cid in data_by_client:
                sorted_data = np.sort(data_by_client[cid])
                cumulative = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
                ax.plot(sorted_data, cumulative, label=f'Client {cid}', 
                       color=color, linewidth=2)
        ax.set_xlabel(sensor_name, fontsize=12)
        ax.set_ylabel('Cumulative Probability', fontsize=12)
        ax.set_title('Cumulative Distribution Function', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.show()
    
    def compute_heterogeneity_metrics(self) -> pd.DataFrame:
        """Compute quantitative heterogeneity metrics"""
        print("\nComputing heterogeneity metrics...")
        
        client_ids = sorted(self.clients.keys())
        metrics = []
        
        # Get common sensors across all clients
        common_sensors = set(self.clients[client_ids[0]].columns)
        for cid in client_ids[1:]:
            common_sensors &= set(self.clients[cid].columns)
        common_sensors = [col for col in common_sensors if col.startswith('sensor_')]
        
        print(f"Common sensors across all clients: {len(common_sensors)}")
        
        # Compute pairwise distances
        for i, cid1 in enumerate(client_ids):
            for cid2 in client_ids[i+1:]:
                # Label distribution distance (JS divergence approximation)
                dist1 = self.clients[cid1]['machine_status'].value_counts(normalize=True)
                dist2 = self.clients[cid2]['machine_status'].value_counts(normalize=True)
                
                # Align distributions
                all_labels = list(set(dist1.index) | set(dist2.index))
                p = np.array([dist1.get(label, 0) for label in all_labels])
                q = np.array([dist2.get(label, 0) for label in all_labels])
                
                # JS divergence
                m = (p + q) / 2
                js_div = 0.5 * np.sum(p * np.log(p / m + 1e-10)) + \
                        0.5 * np.sum(q * np.log(q / m + 1e-10))
                
                # Feature distribution distance (average Wasserstein for common sensors)
                if common_sensors:
                    wasserstein_dists = []
                    ks_stats = []
                    
                    for sensor in common_sensors[:10]:  # Sample 10 sensors
                        data1 = self.clients[cid1][sensor].dropna()
                        data2 = self.clients[cid2][sensor].dropna()
                        
                        if len(data1) > 0 and len(data2) > 0:
                            w_dist = wasserstein_distance(data1, data2)
                            wasserstein_dists.append(w_dist)
                            
                            ks_stat, _ = ks_2samp(data1, data2)
                            ks_stats.append(ks_stat)
                    
                    avg_wasserstein = np.mean(wasserstein_dists) if wasserstein_dists else np.nan
                    avg_ks = np.mean(ks_stats) if ks_stats else np.nan
                else:
                    avg_wasserstein = np.nan
                    avg_ks = np.nan
                
                metrics.append({
                    'Client 1': cid1,
                    'Client 2': cid2,
                    'Label JS Divergence': js_div,
                    'Avg Wasserstein Distance': avg_wasserstein,
                    'Avg KS Statistic': avg_ks
                })
        
        metrics_df = pd.DataFrame(metrics)
        
        print("\nHeterogeneity Metrics Summary:")
        print("="*80)
        print(metrics_df.describe())
        
        return metrics_df
    
    def plot_heterogeneity_heatmap(self, metrics_df: pd.DataFrame = None, save_path: str = None):
        """Plot heatmap of pairwise client heterogeneity"""
        if metrics_df is None:
            metrics_df = self.compute_heterogeneity_metrics()
        
        client_ids = sorted(self.clients.keys())
        n_clients = len(client_ids)
        
        # Create distance matrices
        js_matrix = np.zeros((n_clients, n_clients))
        wasserstein_matrix = np.zeros((n_clients, n_clients))
        ks_matrix = np.zeros((n_clients, n_clients))
        
        for _, row in metrics_df.iterrows():
            i = int(row['Client 1'])
            j = int(row['Client 2'])
            js_matrix[i, j] = js_matrix[j, i] = row['Label JS Divergence']
            wasserstein_matrix[i, j] = wasserstein_matrix[j, i] = row['Avg Wasserstein Distance']
            ks_matrix[i, j] = ks_matrix[j, i] = row['Avg KS Statistic']
        
        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Pairwise Client Heterogeneity Heatmaps', fontsize=16, fontweight='bold')
        
        # JS Divergence
        sns.heatmap(js_matrix, annot=True, fmt='.3f', cmap='YlOrRd',
                   xticklabels=client_ids, yticklabels=client_ids,
                   ax=axes[0], cbar_kws={'label': 'JS Divergence'})
        axes[0].set_title('Label Distribution\n(JS Divergence)', fontsize=13, fontweight='bold')
        axes[0].set_xlabel('Client ID')
        axes[0].set_ylabel('Client ID')
        
        # Wasserstein Distance
        sns.heatmap(wasserstein_matrix, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=client_ids, yticklabels=client_ids,
                   ax=axes[1], cbar_kws={'label': 'Wasserstein Distance'})
        axes[1].set_title('Feature Distribution\n(Wasserstein Distance)', fontsize=13, fontweight='bold')
        axes[1].set_xlabel('Client ID')
        axes[1].set_ylabel('Client ID')
        
        # KS Statistic
        sns.heatmap(ks_matrix, annot=True, fmt='.3f', cmap='Greens',
                   xticklabels=client_ids, yticklabels=client_ids,
                   ax=axes[2], cbar_kws={'label': 'KS Statistic'})
        axes[2].set_title('Statistical Difference\n(KS Statistic)', fontsize=13, fontweight='bold')
        axes[2].set_xlabel('Client ID')
        axes[2].set_ylabel('Client ID')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.show()
    
    def generate_report(self, output_file: str = 'heterogeneity_report.txt'):
        """Generate a text report of heterogeneity analysis"""
        print(f"\nGenerating heterogeneity report...")
        
        with open(output_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("HETEROGENEITY ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Data Directory: {self.data_dir}\n")
            f.write(f"Number of Clients: {len(self.clients)}\n\n")
            
            # Client summaries
            f.write("="*80 + "\n")
            f.write("CLIENT SUMMARIES\n")
            f.write("="*80 + "\n\n")
            
            for cid in sorted(self.clients.keys()):
                client_df = self.clients[cid]
                n_samples = len(client_df)
                n_sensors = len([col for col in client_df.columns if col.startswith('sensor_')])
                missing_rate = client_df.isnull().sum().sum() / (n_samples * len(client_df.columns)) * 100
                
                f.write(f"Client {cid}:\n")
                f.write(f"  Samples: {n_samples:,}\n")
                f.write(f"  Sensors: {n_sensors}\n")
                f.write(f"  Missing Data: {missing_rate:.2f}%\n")
                f.write(f"  Label Distribution:\n")
                
                status_counts = client_df['machine_status'].value_counts()
                for status, count in status_counts.items():
                    pct = count / n_samples * 100
                    f.write(f"    {status}: {count:,} ({pct:.1f}%)\n")
                f.write("\n")
            
            # Heterogeneity metrics
            f.write("="*80 + "\n")
            f.write("HETEROGENEITY METRICS\n")
            f.write("="*80 + "\n\n")
            
            metrics_df = self.compute_heterogeneity_metrics()
            f.write(metrics_df.to_string())
            f.write("\n\n")
            
            f.write("="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")
        
        print(f"Report saved: {output_file}")


def main():
    """Example usage"""
    
    # Visualize hybrid strategy results
    data_dir = 'federated_data/hybrid'
    
    if not os.path.exists(data_dir):
        print(f"Error: {data_dir} not found. Run heterogeneous_data_generator.py first.")
        return
    
    print("="*80)
    print("HETEROGENEITY VISUALIZATION")
    print("="*80)
    
    viz = HeterogeneityVisualizer(data_dir)
    
    # Create output directory for plots
    os.makedirs('plots', exist_ok=True)
    
    # 1. Data distribution
    print("\nGenerating data distribution plots...")
    viz.plot_data_distribution(save_path='plots/data_distribution.png')
    
    # 2. Feature distributions
    print("\nGenerating feature distribution plots...")
    viz.plot_feature_distributions(sensor_name='sensor_00', 
                                  save_path='plots/feature_distribution_sensor00.png')
    
    # 3. Heterogeneity metrics
    print("\nComputing and plotting heterogeneity metrics...")
    metrics_df = viz.compute_heterogeneity_metrics()
    viz.plot_heterogeneity_heatmap(metrics_df, save_path='plots/heterogeneity_heatmap.png')
    
    # 4. Generate report
    viz.generate_report(output_file='plots/heterogeneity_report.txt')
    
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE!")
    print("="*80)
    print("\nGenerated files in plots/:")
    print("  - data_distribution.png")
    print("  - feature_distribution_sensor00.png")
    print("  - heterogeneity_heatmap.png")
    print("  - heterogeneity_report.txt")


if __name__ == "__main__":
    main()

