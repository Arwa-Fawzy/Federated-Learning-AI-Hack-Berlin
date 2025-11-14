# Complete Usage Guide
## Heterogeneous Data Generator for Federated Learning

---

## Table of Contents
1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Detailed Workflow](#detailed-workflow)
4. [Strategy Selection](#strategy-selection)
5. [Parameter Tuning](#parameter-tuning)
6. [Visualization](#visualization)
7. [Federated Learning Integration](#federated-learning-integration)
8. [Advanced Usage](#advanced-usage)
9. [Troubleshooting](#troubleshooting)

---

## Installation

### Step 1: Install Dependencies

```bash
# Option 1: Using requirements.txt
pip install -r requirements.txt

# Option 2: Manual installation
pip install kagglehub pandas numpy scikit-learn matplotlib seaborn scipy pyyaml
```

### Step 2: Download Dataset

```bash
python download_dataset.py
```

This will download the pump sensor data from Kaggle and save it to `data/sensor.csv`.

---

## Quick Start

### Option 1: One-Command Start

```bash
python quick_start.py
```

This will:
- Generate 5 heterogeneous clients (hybrid strategy)
- Create visualizations
- Generate analysis report

### Option 2: Manual Steps

```bash
# 1. Generate clients
python heterogeneous_data_generator.py

# 2. Visualize results
python visualize_heterogeneity.py

# 3. (Optional) Test federated learning
python federated_learning_example.py
```

---

## Detailed Workflow

### Step 1: Explore the Dataset

```bash
python explore_data.py
```

This shows:
- Dataset shape and columns
- Statistical summary
- Missing value analysis
- Label distribution

### Step 2: Generate Heterogeneous Clients

```python
from heterogeneous_data_generator import HeterogeneousDataGenerator

# Initialize
generator = HeterogeneousDataGenerator(
    data_path='data/sensor.csv',
    seed=42
)

# Generate clients
stats = generator.generate_clients(
    n_clients=5,
    strategy='hybrid',
    alpha=0.5,
    output_dir='federated_data/my_experiment'
)
```

### Step 3: Analyze Heterogeneity

```python
from visualize_heterogeneity import HeterogeneityVisualizer

viz = HeterogeneityVisualizer('federated_data/my_experiment')

# Plot distributions
viz.plot_data_distribution(save_path='plots/distribution.png')

# Compute metrics
metrics = viz.compute_heterogeneity_metrics()

# Generate report
viz.generate_report('report.txt')
```

### Step 4: Build Federated Learning System

```python
# See federated_learning_example.py for full implementation
from federated_learning_example import FederatedLearningSimulator

fl_sim = FederatedLearningSimulator('federated_data/my_experiment')
fl_sim.compare_approaches()
```

---

## Strategy Selection

### When to Use Each Strategy

| Strategy | Use Case | Pros | Cons |
|----------|----------|------|------|
| **Clustering** | Different pump types/applications | Realistic operating patterns | May have imbalanced sizes |
| **Dirichlet** | Label skew experiments | Controlled heterogeneity | Doesn't capture feature differences |
| **Temporal** | Seasonal variations | Time-based realism | Sequential, not random |
| **Hybrid** ⭐ | Production-like scenarios | Maximum realism | More complex |

### Strategy Examples

#### 1. Clustering Strategy

Best for: Simulating different industrial facilities with distinct pump configurations.

```python
stats = generator.generate_clients(
    n_clients=5,
    strategy='clustering',
    output_dir='federated_data/clustering'
)
```

**Result**: Clients grouped by operating conditions (high-load, low-load, variable, etc.)

#### 2. Dirichlet Strategy

Best for: Research on label imbalance in federated learning.

```python
# High heterogeneity
stats = generator.generate_clients(
    n_clients=5,
    strategy='dirichlet',
    alpha=0.1,  # Lower = more heterogeneous
    output_dir='federated_data/dirichlet_high'
)

# Low heterogeneity
stats = generator.generate_clients(
    n_clients=5,
    strategy='dirichlet',
    alpha=10.0,  # Higher = more homogeneous
    output_dir='federated_data/dirichlet_low'
)
```

**Result**: Controlled label skew across clients.

#### 3. Temporal Strategy

Best for: Simulating time-based variations (seasonal effects, shift changes).

```python
stats = generator.generate_clients(
    n_clients=4,  # E.g., 4 seasons
    strategy='temporal',
    output_dir='federated_data/temporal'
)
```

**Result**: Clients with data from different time periods.

#### 4. Hybrid Strategy (Recommended)

Best for: Most realistic federated learning scenarios.

```python
stats = generator.generate_clients(
    n_clients=5,
    strategy='hybrid',
    alpha=0.5,
    quantity_distribution=[0.05, 0.35, 0.15, 0.30, 0.15],
    output_dir='federated_data/hybrid'
)
```

**Result**: Combines clustering + label skew + quantity skew + sensor degradation.

---

## Parameter Tuning

### Number of Clients

```python
# Small experiment (fast)
stats = generator.generate_clients(n_clients=3, ...)

# Medium experiment (recommended)
stats = generator.generate_clients(n_clients=5, ...)

# Large experiment (realistic)
stats = generator.generate_clients(n_clients=10, ...)
```

**Trade-offs**:
- More clients → More realistic but slower training
- Fewer clients → Faster but less representative

### Dirichlet Alpha (Heterogeneity Control)

| Alpha | Heterogeneity | Description | Use Case |
|-------|---------------|-------------|----------|
| 0.01-0.1 | **Extreme** | Each client sees 1-2 classes | Research edge cases |
| 0.3-0.7 | **Moderate** | Imbalanced but diverse | Realistic scenarios ⭐ |
| 1.0-5.0 | **Mild** | Slightly imbalanced | Conservative testing |
| >10.0 | **Low** | Nearly balanced | Almost IID baseline |

```python
# Experiment with different alphas
for alpha in [0.1, 0.5, 1.0, 5.0]:
    generator.generate_clients(
        n_clients=5,
        strategy='dirichlet',
        alpha=alpha,
        output_dir=f'federated_data/alpha_{alpha}'
    )
```

### Quantity Distribution

```python
# Equal distribution (IID quantity)
quantity_dist = [0.2, 0.2, 0.2, 0.2, 0.2]

# Pareto-like (realistic, recommended)
quantity_dist = [0.05, 0.35, 0.15, 0.30, 0.15]

# Extreme imbalance
quantity_dist = [0.60, 0.20, 0.10, 0.05, 0.05]

stats = generator.generate_clients(
    n_clients=5,
    strategy='hybrid',
    alpha=0.5,
    quantity_distribution=quantity_dist,
    output_dir='federated_data/custom'
)
```

---

## Visualization

### Available Visualizations

#### 1. Data Distribution

```python
viz.plot_data_distribution(save_path='plots/distribution.png')
```

Shows:
- Sample counts per client
- Label distribution (%)
- Sensor availability
- Missing data rates

#### 2. Feature Distributions

```python
# Compare specific sensor across clients
viz.plot_feature_distributions(
    sensor_name='sensor_00',
    save_path='plots/sensor00.png'
)
```

Shows:
- Histograms (overlapping)
- KDE plots
- Box plots
- Violin plots
- Statistical summary
- Cumulative distribution

#### 3. Heterogeneity Heatmap

```python
metrics = viz.compute_heterogeneity_metrics()
viz.plot_heterogeneity_heatmap(
    metrics,
    save_path='plots/heatmap.png'
)
```

Shows:
- JS Divergence (label distribution)
- Wasserstein Distance (feature distribution)
- KS Statistic (statistical difference)

#### 4. Text Report

```python
viz.generate_report(output_file='report.txt')
```

Contains:
- Client summaries
- Heterogeneity metrics
- Statistical analysis

---

## Federated Learning Integration

### With Flower (Recommended)

```python
import flwr as fl
import pandas as pd
import tensorflow as tf

# Load client data
client_data = pd.read_csv('federated_data/hybrid/client_0.csv')

# Define Flower client
class PumpSensorClient(fl.client.NumPyClient):
    def __init__(self, client_id):
        self.client_id = client_id
        self.model = create_model()
        self.data = load_client_data(client_id)
    
    def get_parameters(self, config):
        return self.model.get_weights()
    
    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(self.data['X_train'], self.data['y_train'], epochs=5)
        return self.model.get_weights(), len(self.data['X_train']), {}
    
    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.data['X_test'], self.data['y_test'])
        return loss, len(self.data['X_test']), {"accuracy": accuracy}

# Start Flower client
fl.client.start_numpy_client(
    server_address="localhost:8080",
    client=PumpSensorClient(client_id=0)
)
```

### With TensorFlow Federated

```python
import tensorflow_federated as tff

# Load all client data
client_datasets = []
for i in range(5):
    data = pd.read_csv(f'federated_data/hybrid/client_{i}.csv')
    client_datasets.append(preprocess(data))

# Define TFF model
def model_fn():
    return tff.learning.from_keras_model(
        keras_model=create_keras_model(),
        input_spec=input_spec,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )

# Build iterative process
iterative_process = tff.learning.build_federated_averaging_process(model_fn)

# Initialize
state = iterative_process.initialize()

# Training rounds
for round_num in range(100):
    state, metrics = iterative_process.next(state, client_datasets)
    print(f'Round {round_num}: {metrics}')
```

### Simple Custom Implementation

```python
# See federated_learning_example.py for complete code

# 1. Load client datasets
clients = {}
for i in range(5):
    clients[i] = pd.read_csv(f'federated_data/hybrid/client_{i}.csv')

# 2. Define local training
def train_local(client_data, global_weights):
    model = create_model()
    model.set_weights(global_weights)
    model.fit(X_train, y_train, epochs=5)
    return model.get_weights()

# 3. Federated averaging
def fed_avg(client_weights):
    return np.mean(client_weights, axis=0)

# 4. Training loop
global_weights = initialize_model().get_weights()

for round in range(100):
    client_weights = []
    
    # Local training
    for client_id, data in clients.items():
        weights = train_local(data, global_weights)
        client_weights.append(weights)
    
    # Aggregate
    global_weights = fed_avg(client_weights)
    
    # Evaluate
    evaluate_global_model(global_weights)
```

---

## Advanced Usage

### Custom Sensor Degradation

```python
# Modify the generator to add custom degradation
generator = HeterogeneousDataGenerator('data/sensor.csv')

# Override degradation method
def custom_degradation(client_df):
    # Your custom degradation logic
    return client_df

# Apply to clients
clients = generator.generate_clients(...)
```

### Multi-Experiment Setup

```python
# Run multiple experiments with different configurations
experiments = [
    {'name': 'low_het', 'strategy': 'dirichlet', 'alpha': 5.0},
    {'name': 'med_het', 'strategy': 'dirichlet', 'alpha': 0.5},
    {'name': 'high_het', 'strategy': 'dirichlet', 'alpha': 0.1},
    {'name': 'hybrid', 'strategy': 'hybrid', 'alpha': 0.5},
]

for exp in experiments:
    stats = generator.generate_clients(
        n_clients=5,
        strategy=exp['strategy'],
        alpha=exp['alpha'],
        output_dir=f"federated_data/{exp['name']}"
    )
```

### Batch Visualization

```python
# Visualize all experiments
experiment_dirs = [
    'federated_data/low_het',
    'federated_data/med_het',
    'federated_data/high_het',
    'federated_data/hybrid'
]

for exp_dir in experiment_dirs:
    viz = HeterogeneityVisualizer(exp_dir)
    exp_name = exp_dir.split('/')[-1]
    
    viz.plot_data_distribution(save_path=f'plots/{exp_name}_dist.png')
    viz.generate_report(f'plots/{exp_name}_report.txt')
```

---

## Troubleshooting

### Common Issues

#### 1. "Module not found: kagglehub"

```bash
pip install kagglehub
```

#### 2. "File not found: data/sensor.csv"

```bash
python download_dataset.py
```

#### 3. Memory Error

```python
# Reduce number of clients
generator.generate_clients(n_clients=3, ...)

# Or use sampling
df_sample = df.sample(frac=0.5)  # Use 50% of data
```

#### 4. Plots not displaying

```python
import matplotlib
matplotlib.use('TkAgg')  # Or 'Qt5Agg'
import matplotlib.pyplot as plt
```

#### 5. Slow clustering

```python
# Reduce number of sensors used for clustering
# Edit heterogeneous_data_generator.py line ~123
clustering_sensors = [col for col in self.sensor_cols[:10]]  # Use only 10 sensors
```

---

## Performance Tips

1. **Use hybrid strategy** for most realistic results
2. **Start with α=0.5** for moderate heterogeneity
3. **Use 5 clients** as a good default
4. **Enable visualization** to understand data distribution
5. **Test with small experiments** before large-scale runs
6. **Use random seed** for reproducibility

---

## Best Practices

### For Research
- Run multiple seeds to ensure reproducibility
- Test various α values systematically
- Document heterogeneity metrics
- Compare with IID baseline

### For Production
- Use hybrid strategy for realism
- Monitor heterogeneity metrics over time
- Implement proper FL algorithms (FedAvg, FedProx)
- Add differential privacy for security

### For Teaching
- Start with clustering (easiest to understand)
- Visualize before training
- Compare centralized vs federated results
- Discuss privacy trade-offs

---

## Additional Resources

- [Federated Learning Paper](https://arxiv.org/abs/1602.05629)
- [Flower Documentation](https://flower.dev/)
- [TensorFlow Federated Guide](https://www.tensorflow.org/federated)
- [Non-IID Data in FL](https://arxiv.org/abs/1909.06335)

---

**Questions?** Refer to README.md or open an issue.

