# Heterogeneous Data Generator for Federated Learning
## Pump Sensor Data - Federated Learning Setup

This project provides tools to create realistic heterogeneous datasets from pump sensor data for federated learning experiments. It simulates multiple facilities/clients with different operating conditions, sensor configurations, and failure patterns.

---

## 🎯 Overview

**Problem**: Real-world federated learning involves clients with:
- Different data distributions (non-IID)
- Varying sensor quality and configurations
- Unequal dataset sizes
- Different failure modes

**Solution**: This toolkit generates heterogeneous client datasets from a single-location pump sensor dataset using multiple strategies.

---

## 📦 Features

### 1. **Multiple Partitioning Strategies**
- **Clustering**: Partition by operating conditions using K-means
- **Dirichlet**: Create label imbalance using Dirichlet distribution
- **Temporal**: Split by time periods for seasonal variations
- **Hybrid**: Combine all strategies for maximum realism (recommended)

### 2. **Heterogeneity Mechanisms**
- **Operating Condition Diversity**: Different pump workloads and patterns
- **Label Distribution Skew**: Imbalanced failure modes across clients
- **Sensor Quality Degradation**: Calibration bias, noise, dropout, delays
- **Feature Subset Heterogeneity**: Different sensor configurations
- **Quantity Skew**: Realistic Pareto-like data distribution

### 3. **Comprehensive Visualization**
- Data distribution analysis
- Feature distribution comparisons
- Heterogeneity heatmaps
- Statistical metrics (JS Divergence, Wasserstein Distance, KS Statistic)
- Automated reporting

---

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install kagglehub pandas numpy scikit-learn matplotlib seaborn scipy

# Download the dataset
python download_dataset.py
```

### Generate Heterogeneous Clients

```bash
# Run with default settings (hybrid strategy)
python heterogeneous_data_generator.py
```

This will create:
- `federated_data/hybrid/` - 5 heterogeneous client datasets
- `federated_data/clustering/` - Pure clustering approach
- `federated_data/dirichlet_high/` - High heterogeneity via Dirichlet

### Visualize Results

```bash
# Generate plots and analysis
python visualize_heterogeneity.py
```

Output in `plots/`:
- `data_distribution.png` - Client dataset characteristics
- `feature_distribution_sensor00.png` - Feature distribution analysis
- `heterogeneity_heatmap.png` - Pairwise heterogeneity metrics
- `heterogeneity_report.txt` - Detailed text report

---

## 📚 Usage

### Basic Usage

```python
from heterogeneous_data_generator import HeterogeneousDataGenerator

# Initialize
generator = HeterogeneousDataGenerator(
    data_path='data/sensor.csv',
    seed=42
)

# Generate clients (hybrid strategy)
stats = generator.generate_clients(
    n_clients=5,
    strategy='hybrid',
    alpha=0.5,
    output_dir='federated_data/my_experiment'
)
```

### Advanced Configuration

```python
# Custom quantity distribution (Pareto-like)
quantity_dist = [0.05, 0.35, 0.15, 0.30, 0.15]  # Must sum to 1.0

stats = generator.generate_clients(
    n_clients=5,
    strategy='hybrid',
    alpha=0.3,  # Higher heterogeneity
    quantity_distribution=quantity_dist,
    output_dir='federated_data/custom'
)
```

### Visualization

```python
from visualize_heterogeneity import HeterogeneityVisualizer

viz = HeterogeneityVisualizer('federated_data/hybrid')

# Plot data distribution
viz.plot_data_distribution(save_path='plots/distribution.png')

# Plot feature distributions
viz.plot_feature_distributions(
    sensor_name='sensor_00',
    save_path='plots/sensor00.png'
)

# Compute heterogeneity metrics
metrics = viz.compute_heterogeneity_metrics()

# Generate heatmaps
viz.plot_heterogeneity_heatmap(metrics, save_path='plots/heatmap.png')

# Generate report
viz.generate_report(output_file='report.txt')
```

---

## 🔧 Strategies Explained

### 1. **Clustering Strategy**

Groups data by operating conditions using K-means clustering.

```python
generator.generate_clients(
    n_clients=5,
    strategy='clustering',
    output_dir='federated_data/clustering'
)
```

**Best for**: Simulating facilities with distinct pump types or applications

### 2. **Dirichlet Strategy**

Creates label imbalance using Dirichlet distribution.

```python
generator.generate_clients(
    n_clients=5,
    strategy='dirichlet',
    alpha=0.1,  # Lower = more heterogeneous
    output_dir='federated_data/dirichlet'
)
```

**Best for**: Testing federated learning robustness to label skew

**Alpha parameter**:
- `alpha=0.1`: Extreme heterogeneity (each client sees 1-2 classes)
- `alpha=0.5`: Moderate heterogeneity (recommended)
- `alpha=1.0`: Mild heterogeneity
- `alpha=10.0`: Nearly homogeneous

### 3. **Temporal Strategy**

Splits data by time periods for seasonal variations.

```python
generator.generate_clients(
    n_clients=5,
    strategy='temporal',
    output_dir='federated_data/temporal'
)
```

**Best for**: Simulating seasonal effects or time-based differences

### 4. **Hybrid Strategy** ⭐ (Recommended)

Combines clustering + Dirichlet + quantity skew for maximum realism.

```python
generator.generate_clients(
    n_clients=5,
    strategy='hybrid',
    alpha=0.5,
    output_dir='federated_data/hybrid'
)
```

**Best for**: Production-like federated learning scenarios

---

## 📊 Dataset Structure

### Original Dataset
- **Source**: [Kaggle - Pump Sensor Data](https://www.kaggle.com/datasets/nphantawee/pump-sensor-data)
- **Size**: 220,320 samples × 55 columns
- **Features**: 52 sensors (temperature, pressure, vibration, flow, RPM, etc.)
- **Target**: `machine_status` (NORMAL, RECOVERING, BROKEN)
- **Time Range**: April - August 2018

### Generated Client Datasets

Each client CSV contains:
- **Timestamp**: Time series data
- **Sensors**: Subset of original sensors (varies by client)
- **machine_status**: Target label
- **Artifacts**: Applied degradation (bias, noise, dropout, delay)

### Metadata (`client_metadata.json`)

```json
{
  "n_clients": 5,
  "total_samples": 220320,
  "clients": {
    "0": {
      "samples": 11016,
      "sensors": 52,
      "status_distribution": {
        "NORMAL": 10245,
        "RECOVERING": 771,
        "BROKEN": 0
      },
      "missing_rate": 0.0234,
      "file": "client_0.csv"
    },
    ...
  }
}
```

---

## 📈 Heterogeneity Metrics

### 1. **JS Divergence** (Label Distribution)
Measures difference in label distributions between clients.
- Range: [0, ∞)
- 0 = identical distributions
- Higher = more heterogeneous

### 2. **Wasserstein Distance** (Feature Distribution)
Earth Mover's Distance for feature distributions.
- Range: [0, ∞)
- Captures shift and spread differences

### 3. **KS Statistic** (Statistical Difference)
Kolmogorov-Smirnov test statistic.
- Range: [0, 1]
- >0.05 typically indicates significant difference

---

## 🛠️ Configuration

Edit `config.yaml` to customize:
- Number of clients
- Strategy and parameters
- Heterogeneity options
- Visualization settings

```yaml
federated_learning:
  num_clients: 5
  strategy: "hybrid"
  dirichlet_alpha: 0.5
  output_dir: "federated_data/custom"
```

---

## 📁 Project Structure

```
dist-hack/
├── data/
│   └── sensor.csv                      # Downloaded dataset
├── federated_data/
│   ├── hybrid/                          # Generated clients (hybrid)
│   │   ├── client_0.csv
│   │   ├── client_1.csv
│   │   ├── ...
│   │   └── client_metadata.json
│   ├── clustering/                      # Clustering strategy
│   └── dirichlet_high/                  # Dirichlet strategy
├── plots/
│   ├── data_distribution.png
│   ├── feature_distribution_sensor00.png
│   ├── heterogeneity_heatmap.png
│   └── heterogeneity_report.txt
├── download_dataset.py                  # Download from Kaggle
├── explore_data.py                      # Dataset exploration
├── heterogeneous_data_generator.py      # Main generator
├── visualize_heterogeneity.py           # Visualization tools
├── config.yaml                          # Configuration
└── README.md                            # This file
```

---

## 🧪 Example Use Cases

### 1. **Federated Predictive Maintenance**

Train a model to predict pump failures across multiple facilities without sharing sensitive operational data.

```python
# Generate realistic client data
generator.generate_clients(
    n_clients=10,
    strategy='hybrid',
    alpha=0.3,  # Moderate heterogeneity
    output_dir='federated_data/predictive_maintenance'
)

# Each facility trains locally on their data
# Central server aggregates model updates (FedAvg)
```

### 2. **Anomaly Detection**

Detect unusual pump behavior across facilities with different normal operating conditions.

```python
# High heterogeneity to test robustness
generator.generate_clients(
    n_clients=5,
    strategy='clustering',  # Group by operating patterns
    output_dir='federated_data/anomaly_detection'
)
```

### 3. **Research: Non-IID Data**

Study impact of heterogeneity on federated learning algorithms.

```python
# Generate multiple datasets with varying heterogeneity
for alpha in [0.1, 0.5, 1.0, 10.0]:
    generator.generate_clients(
        n_clients=5,
        strategy='dirichlet',
        alpha=alpha,
        output_dir=f'federated_data/alpha_{alpha}'
    )
```

---

## 🔬 Heterogeneity Sources

The generator creates heterogeneity through:

1. **Operating Conditions** (Clustering)
   - High-load vs low-load operations
   - Continuous vs start-stop cycles
   - Different pump applications

2. **Label Distribution** (Dirichlet)
   - New facility (mostly NORMAL)
   - Aging facility (more RECOVERING/BROKEN)
   - Balanced mix

3. **Sensor Quality** (Degradation)
   - Calibration bias (±5-10%)
   - Measurement noise (10% of std)
   - Intermittent dropout (15% missing)
   - Signal delay (1-5 timesteps)

4. **Sensor Configuration** (Feature Subsets)
   - Full suite (100% sensors)
   - High-end (90% sensors)
   - Standard (70% sensors)
   - Budget (50% sensors)
   - Legacy (40% sensors)

5. **Dataset Size** (Quantity Skew)
   - Small facility (5% of data)
   - Large facility (35% of data)
   - Pareto-like distribution

---

## 🎓 Federated Learning Integration

### Recommended Frameworks

1. **Flower (flwr)** - Modern, flexible
   ```bash
   pip install flwr
   ```

2. **TensorFlow Federated** - Google's framework
   ```bash
   pip install tensorflow-federated
   ```

3. **PySyft** - Privacy-focused
   ```bash
   pip install syft
   ```

### Example FL Workflow

```python
# 1. Generate heterogeneous clients
generator.generate_clients(n_clients=5, strategy='hybrid')

# 2. Load client data
clients = {}
for i in range(5):
    clients[i] = pd.read_csv(f'federated_data/hybrid/client_{i}.csv')

# 3. Define local training function
def train_local_model(client_data, global_weights):
    # Train on local data
    model = create_model()
    model.set_weights(global_weights)
    model.fit(X_train, y_train, epochs=5)
    return model.get_weights()

# 4. Federated averaging
def federated_averaging(client_weights):
    # Average weights from all clients
    return np.mean(client_weights, axis=0)

# 5. Iterate for N rounds
for round in range(100):
    client_weights = []
    for client_id, data in clients.items():
        weights = train_local_model(data, global_weights)
        client_weights.append(weights)
    
    global_weights = federated_averaging(client_weights)
```

---

## 📊 Expected Results

### Data Distribution
- Client 0: 5% of data (11,016 samples) - Small facility
- Client 1: 35% of data (77,112 samples) - Large facility
- Client 2-4: 15-30% of data - Medium facilities

### Label Imbalance
- Varies by strategy and alpha parameter
- Hybrid: Moderate imbalance across clients
- Dirichlet (α=0.1): Extreme imbalance

### Sensor Availability
- Client 0: All 52 sensors
- Client 1: 45 sensors (High-end)
- Client 2: 35 sensors (Standard)
- Client 3: 25 sensors (Budget)
- Client 4: 20 sensors (Legacy)

### Missing Data Rates
- Varies by degradation applied (0-15% missing)

---

## 🐛 Troubleshooting

### Issue: "Module not found: kagglehub"
```bash
pip install kagglehub
```

### Issue: "File not found: data/sensor.csv"
```bash
python download_dataset.py
```

### Issue: Unicode encoding error
- Fixed in latest version (replaced ✓ with [OK])

### Issue: Visualization plots not showing
```bash
# Install matplotlib backend
pip install matplotlib --upgrade
```

### Issue: Out of memory
```python
# Reduce number of clients or use fewer samples
generator.generate_clients(n_clients=3, ...)
```

---

## 📝 Citation

If you use this tool in your research, please cite:

```bibtex
@misc{heterogeneous_pump_data_2024,
  title={Heterogeneous Data Generator for Federated Learning on Pump Sensor Data},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/dist-hack}
}
```

Original dataset:
```bibtex
@dataset{pump_sensor_data,
  title={Pump Sensor Data},
  author={Phantawee, N.},
  year={2018},
  url={https://www.kaggle.com/datasets/nphantawee/pump-sensor-data}
}
```

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🔗 References

- [Federated Learning](https://arxiv.org/abs/1602.05629) - McMahan et al., 2017
- [FedAvg Algorithm](https://arxiv.org/abs/1602.05629)
- [FedProx](https://arxiv.org/abs/1812.06127) - For non-IID data
- [Dirichlet Distribution for FL](https://arxiv.org/abs/1909.06335)
- [Flower Framework](https://flower.dev/)

---

## 💡 Tips for Best Results

1. **Start with hybrid strategy** - Most realistic scenario
2. **Use α=0.5** for moderate heterogeneity - Good balance
3. **Visualize before training** - Understand your data distribution
4. **Monitor heterogeneity metrics** - Track JS divergence, Wasserstein distance
5. **Test with multiple seeds** - Ensure reproducibility
6. **Consider FedProx** for highly non-IID data

---

## 🎯 Next Steps

After generating heterogeneous data:

1. **Explore the data**: Run visualization scripts
2. **Build FL system**: Integrate with Flower/TensorFlow Federated
3. **Train models**: Implement local training + aggregation
4. **Compare strategies**: Centralized vs Federated
5. **Optimize**: Try FedAvg, FedProx, FedAdam
6. **Add privacy**: Differential privacy, secure aggregation

---

**Questions?** Open an issue or contact the maintainer.

**Happy Federated Learning! 🚀**

