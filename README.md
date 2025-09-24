# Federated Learning Algorithms Comparison

This repository contains implementations and experimental results for comparing various federated learning algorithms on CIFAR-10 and CIFAR-100 datasets under different data heterogeneity settings.

---

## 🔍 Overview

This project implements and compares four state-of-the-art federated learning algorithms:
- **Scaffnew**: Enhanced SCAFFOLD algorithm with improved convergence properties
- **FedLin**: Linear federated optimization method
- **SCAFFOLD**: Communication-efficient federated learning with variance reduction
- **Fedrcu**: Federated learning with recursive control updates

The experiments evaluate these algorithms under various degrees of data heterogeneity using Dirichlet distribution sampling.

---

## 📁 Project Structure

```
o1_t/
├── main.py                 # Main experiment execution script
├── trainers.py             # Federated learning algorithm implementations
├── models.py               # Neural network model definitions
├── optimizers.py           # Custom optimizers for federated learning
├── CNN_CGT.py             # CNN model with Compressed Gradient Transfer
├── plot.py                 # Visualization utilities
├── test.py                 # Testing utilities
├── data/                   # Dataset storage directory
├── results/                # Experimental results
│   ├── cifar10_e3000_homFalse_0_L_2_dir_0.1/
│   ├── cifar10_e3000_homFalse_0_L_2_dir_1/
│   ├── cifar10_e3000_homFalse_0_L_2_dir_10/
│   ├── cifar100_e3000_homFalse_0_L_2_dir_0.1/
│   ├── cifar100_e3000_homFalse_0_L_2_dir_1/
│   └── cifar100_e3000_homFalse_0_L_2_dir_10/
└── __pycache__/           # Python cache files
```

---

## 🚀 Algorithms Implemented

### 1. Scaffnew Trainer
Enhanced version of SCAFFOLD with improved theoretical guarantees and practical performance.

### 2. FedLin Trainer  
Linear federated optimization method designed for efficient communication and convergence.

### 3. SCAFFOLD Trainer
Variance reduction technique that uses control variates to reduce communication complexity.

### 4. Fedrcu Trainer
Federated learning with recursive control updates for better handling of heterogeneous data.

---

## 📊 Experimental Setup

### Datasets
- **CIFAR-10**: 10-class image classification (32×32 RGB images)
- **CIFAR-100**: 100-class image classification (32×32 RGB images)

### Configuration Parameters
- **Agents**: 10 federated clients
- **Communication Rounds**: 3000
- **Local Steps**: 10 per communication round
- **Batch Size**: 128
- **Smoothness Parameter (L)**: 2

### Data Heterogeneity Levels
The experiments use Dirichlet distribution to create non-IID data splits:
- **dir_0.1**: Highly heterogeneous (α = 0.1)
- **dir_1**: Moderately heterogeneous (α = 1.0) 
- **dir_10**: Mildly heterogeneous (α = 10.0)

---

## 🛠️ Usage

### Running Experiments

Execute the main experiment script:
```bash
python main.py
```

This will run all algorithms on the configured dataset with specified parameters and save results to the `results/` directory.

### Customizing Experiments

Modify parameters in `main.py`:
```python
agents = 10                    # Number of federated clients
dataset = "cifar100"          # "cifar10" or "cifar100"
communication_round = 3000    # Number of communication rounds
local_steps = 10             # Local updates per round
bs = 128                     # Batch size
L = 2                        # Smoothness parameter
dir_alpha = 10               # Dirichlet concentration parameter
```

### Generating Visualizations

Use the plotting utility to generate comparison charts:
```bash
python plot.py
```

---

## 📈 Results Format

Each experiment generates CSV files with the following structure:
```csv
"Algorithm Name, hyperparameters"
step0,step1,step2,...,stepN
train_accuracy_values
test_accuracy_values  
loss_values
```

Results are organized by:
- Dataset (CIFAR-10/CIFAR-100)
- Communication rounds (e3000)
- Heterogeneity setting (dir_0.1, dir_1, dir_10)
- Algorithm-specific parameters

---

## 📋 Dependencies

```bash
pip install torch torchvision numpy matplotlib pandas
```

Required packages:
- **PyTorch**: Deep learning framework
- **torchvision**: Computer vision datasets and transforms
- **numpy**: Numerical computing
- **matplotlib**: Plotting and visualization
- **pandas**: Data manipulation and analysis

---

## 🔧 Key Features

- **Modular Design**: Easy to add new federated learning algorithms
- **Flexible Data Distribution**: Support for various non-IID settings
- **Comprehensive Evaluation**: Training accuracy, test accuracy, and loss tracking
- **Reproducible Experiments**: Fixed random seeds and consistent experimental setup
- **Efficient Implementation**: Optimized for multi-client federated learning scenarios

---

## 📊 Performance Metrics

The experiments track three key metrics:
1. **Training Accuracy**: Performance on local training data
2. **Test Accuracy**: Generalization performance on held-out test data
3. **Loss Values**: Convergence behavior over communication rounds

---

## 🤝 Contributing

We welcome contributions! Please consider:
- Adding new federated learning algorithms
- Implementing additional datasets
- Improving visualization capabilities
- Enhancing documentation

---
