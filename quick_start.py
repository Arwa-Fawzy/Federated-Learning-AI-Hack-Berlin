"""
Quick Start Example for Heterogeneous Data Generator
Run this script to quickly generate and visualize heterogeneous federated data
"""

import os
from heterogeneous_data_generator import HeterogeneousDataGenerator
from visualize_heterogeneity import HeterogeneityVisualizer


def main():
    print("="*80)
    print("HETEROGENEOUS FEDERATED LEARNING DATA GENERATOR - QUICK START")
    print("="*80)
    
    # Check if data exists
    data_path = 'data/sensor.csv'
    if not os.path.exists(data_path):
        print("\n❌ Error: Dataset not found!")
        print("Please run: python download_dataset.py")
        return
    
    print("\n✓ Dataset found!")
    
    # Step 1: Generate heterogeneous clients
    print("\n" + "="*80)
    print("STEP 1: Generating Heterogeneous Clients")
    print("="*80)
    
    generator = HeterogeneousDataGenerator(
        data_path=data_path,
        seed=42
    )
    
    # Generate with hybrid strategy (recommended)
    print("\nGenerating 5 clients with HYBRID strategy...")
    stats = generator.generate_clients(
        n_clients=5,
        strategy='hybrid',
        alpha=0.5,  # Moderate heterogeneity
        output_dir='federated_data/quickstart'
    )
    
    print("\n✓ Client generation complete!")
    print(f"  Total samples: {stats['total_samples']:,}")
    print(f"  Clients: {stats['n_clients']}")
    
    # Step 2: Visualize heterogeneity
    print("\n" + "="*80)
    print("STEP 2: Visualizing Heterogeneity")
    print("="*80)
    
    viz = HeterogeneityVisualizer('federated_data/quickstart')
    
    # Create plots directory
    os.makedirs('plots/quickstart', exist_ok=True)
    
    print("\nGenerating visualizations...")
    
    # Data distribution
    print("  1. Data distribution plot...")
    viz.plot_data_distribution(save_path='plots/quickstart/data_distribution.png')
    
    # Feature distributions
    print("  2. Feature distribution plot...")
    viz.plot_feature_distributions(
        sensor_name='sensor_00',
        save_path='plots/quickstart/feature_distribution.png'
    )
    
    # Heterogeneity metrics
    print("  3. Heterogeneity heatmap...")
    metrics = viz.compute_heterogeneity_metrics()
    viz.plot_heterogeneity_heatmap(
        metrics,
        save_path='plots/quickstart/heterogeneity_heatmap.png'
    )
    
    # Report
    print("  4. Generating report...")
    viz.generate_report(output_file='plots/quickstart/heterogeneity_report.txt')
    
    print("\n✓ Visualization complete!")
    
    # Step 3: Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print("\n📁 Generated Files:")
    print("\n  Client Data:")
    print("    - federated_data/quickstart/client_0.csv")
    print("    - federated_data/quickstart/client_1.csv")
    print("    - federated_data/quickstart/client_2.csv")
    print("    - federated_data/quickstart/client_3.csv")
    print("    - federated_data/quickstart/client_4.csv")
    print("    - federated_data/quickstart/client_metadata.json")
    
    print("\n  Visualizations:")
    print("    - plots/quickstart/data_distribution.png")
    print("    - plots/quickstart/feature_distribution.png")
    print("    - plots/quickstart/heterogeneity_heatmap.png")
    print("    - plots/quickstart/heterogeneity_report.txt")
    
    print("\n" + "="*80)
    print("✅ QUICK START COMPLETE!")
    print("="*80)
    
    print("\n📖 Next Steps:")
    print("  1. Explore the generated client datasets")
    print("  2. Review the visualization plots")
    print("  3. Read heterogeneity_report.txt for detailed metrics")
    print("  4. Build your federated learning system using these clients")
    print("  5. Check README.md for advanced usage and FL integration")
    
    print("\n🚀 Ready for Federated Learning!")
    print("\nTo customize generation, edit config.yaml or use:")
    print("  python heterogeneous_data_generator.py")
    print("\nTo visualize different datasets, use:")
    print("  python visualize_heterogeneity.py")
    print("\n")


if __name__ == "__main__":
    main()

