# FedLin2 - Federated Learning Experiments

This repository contains code and results for federated learning experiments comparing different algorithms on CIFAR-10 and CIFAR-100 datasets.

## Project Structure

```
fedlin2/
├── CNN_CGT.py              # CNN model with CGT (Compressed Gradient Transfer)
├── main.py                 # Main execution script
├── models.py               # Neural network model definitions
├── optimizers.py           # Custom optimizers for federated learning
├── plot.py                 # Plotting utilities for result visualization
├── test.py                 # Testing utilities
├── trainers.py             # Training algorithms and federated learning methods
├── data/                   # Dataset storage
│   ├── cifar-10-python.tar.gz
│   ├── cifar-100-python.tar.gz
│   └── [extracted datasets]
├── results/                # Experimental results
│   ├── cifar10_e3000_homFalse_0_L_2_dir_0.1/
│   ├── cifar10_e3000_homFalse_0_L_2_dir_1/
│   ├── cifar10_e3000_homFalse_0_L_2_dir_10/
│   ├── cifar100_e3000_homFalse_0_L_2_dir_0.1/
│   ├── cifar100_e3000_homFalse_0_L_2_dir_1/
│   └── cifar100_e3000_homFalse_0_L_2_dir_10/
├── save/                   # Saved model checkpoints and additional results
└── fedlin2_eps/           # Generated EPS plots (archived)
```

## Algorithms Compared

The experiments compare four federated learning algorithms:

- **FedAvg** (Black) - Federated Averaging
- **FedProx** (Red) - Federated Proximal
- **FedNova** (Orange) - Federated Nova
- **Scaffnew** (Blue) - SCAFFOLD with improvements

## Datasets and Configurations

### Datasets
- **CIFAR-10**: 10-class image classification
- **CIFAR-100**: 100-class image classification

### Experimental Settings
- **Epochs**: 3000 communication rounds
- **Non-IID Distribution**: Dirichlet distribution with different concentration parameters
  - `dir_0.1`: Highly non-IID (α = 0.1)
  - `dir_1`: Moderately non-IID (α = 1.0)
  - `dir_10`: Mildly non-IID (α = 10.0)

## Usage

### Running Experiments
```bash
python main.py
```

### Generating Plots
The `plot.py` script automatically generates visualization plots for experimental results:

```bash
# Generate plots for a specific experiment
python plot.py /path/to/experiment.csv

# Generate plots for all experiments
for csv_file in $(find results -name "*.csv"); do
    python plot.py "$csv_file"
done
```

### Plot Types Generated
Each experiment generates three types of plots:
1. **Train Accuracy Comparison** - Training accuracy over communication rounds
2. **Loss Comparison** - Loss values over communication rounds  
3. **Test Accuracy Comparison** - Test accuracy over communication rounds

## Plot Features

- **Consistent Color Scheme**: Each algorithm has a fixed color across all plots
- **High-Quality Output**: Supports both PNG (for preview) and EPS (for publication)
- **Publication Ready**: Clean styling with proper labels and legends
- **Automatic Processing**: Batch processing of multiple experiments

### Color Mapping
- **Black**: Scaffnew
- **Red**: Algorithm2
- **Orange**: SCAFFOLD
- **Blue**: FedLin2/FedLin

## Results Structure

Each experiment directory contains:
- `{experiment_name}.csv` - Raw experimental data
- `train_accuracy_comparison_{dataset}_dir{α}.png` - Training accuracy plot
- `loss_comparison_{dataset}_dir{α}.png` - Loss progression plot
- `test_accuracy_comparison_{dataset}_dir{α}.png` - Test accuracy plot

## File Format

The CSV files use a block format where each block contains:
```
"Method, hyperparameters"
step0,step1,step2,...
train_accuracy_values
test_accuracy_values
loss_values
```

## Dependencies

- Python 3.x
- matplotlib
- numpy
- [Other dependencies as needed]

## Configuration

Key parameters can be adjusted in the respective files:
- Model architecture: `models.py`
- Training algorithms: `trainers.py`
- Optimization settings: `optimizers.py`
- Plotting preferences: `plot.py`

## Citation

If you use this code in your research, please cite:
```
[Add appropriate citation information]
```

## License

[Add license information]

## Contact

[Add contact information]# o1_t
