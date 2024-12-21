# Deep Learning Project: Neural Network Implementations

**Author:** Anurag Mishra  
**Course:** CISC 865 Deep Learning  
**Final Project**

## Overview
This project explores and compares two neural network paradigms:
1. **Bayesian Neural Network (BNN):** A convolutional neural network with uncertainty estimation using Monte Carlo Dropout.
2. **Hopfield Network:** A classical associative memory network for pattern retrieval.

The analysis highlights the strengths and limitations of both approaches, focusing on tasks like uncertainty quantification and robust pattern retrieval.

## Project Structure
```
.
├── bayesian_nn/
│   ├── evaluate.py         # Evaluation script with uncertainty analysis
│   ├── model.py            # Bayesian CNN implementation
│   └── train.py            # Training script for Bayesian CNN
├── hopfield/
│   ├── hopfield_network.py # Hopfield Network implementation
│   └── run_hopfield.py     # Demonstration and analysis of Hopfield Networks
├── results/
│   ├── bayesian_nn_results/
│   │   ├── calibration.png     # Calibration results
│   │   ├── ood_analysis.png    # Out-of-distribution analysis
│   │   └── uncertainty_analysis.png # Uncertainty vs. noise analysis
│   ├── hopfield_results/
│   │   ├── basic_retrieval.png     # Example of pattern retrieval
│   │   ├── basin_of_attraction.png # Basin of attraction analysis
│   │   ├── capacity_analysis.png   # Capacity analysis
│   │   ├── energy_convergence.png  # Energy convergence during recall
│   │   └── pattern_overlap.png     # Pattern overlap analysis
├── requirements.txt        # Python dependencies for the project
└── README.md               # Project documentation
```

## Requirements
- Python 3.8+
- PyTorch 1.12+
- torchvision
- NumPy
- Matplotlib
- Seaborn

Install dependencies with:
```bash
pip install -r requirements.txt
```

## Usage
### Bayesian Neural Network
1. **Training**
   Train the BNN using the `train.py` script:
   ```bash
   python bayesian_nn/train.py
   ```
   This trains the model on a subset of the MNIST dataset and saves the model weights.

2. **Evaluation**
   Evaluate the BNN with uncertainty quantification using `evaluate.py`:
   ```bash
   python bayesian_nn/evaluate.py
   ```
   This generates results for:
   - Uncertainty analysis vs. noise levels
   - Out-of-distribution detection
   - Calibration analysis

### Hopfield Network
1. **Pattern Storage and Retrieval**
   Run `run_hopfield.py` to demonstrate pattern retrieval, analyze energy convergence, and assess capacity:
   ```bash
   python hopfield/run_hopfield.py
   ```
   This script produces visualizations for:
   - Basic pattern retrieval
   - Energy convergence
   - Basin of attraction analysis
   - Capacity analysis

2. **Customization**
   Modify patterns and noise levels in `run_hopfield.py` for custom experiments.

## Key Results
- **Hopfield Network:** Demonstrated robust pattern retrieval up to its theoretical capacity (~8 patterns for 64 neurons).
  - Key visualizations:
    - `basic_retrieval.png`: Demonstrates pattern restoration from noisy input.
    - `capacity_analysis.png`: Validates theoretical capacity.
    - `energy_convergence.png`: Tracks energy minimization during retrieval.
- **Bayesian Neural Network:** Showcased well-calibrated predictions and sensitivity to out-of-distribution inputs.
  - Key visualizations:
    - `calibration.png`: Reliability diagram (ECE = 0.0269).
    - `ood_analysis.png`: Distinguishes in-distribution vs. OOD uncertainties.
    - `uncertainty_analysis.png`: Captures uncertainty variations with noise levels.

## Future Directions
- **Hybrid Models:** Explore integrating Hopfield Networks with BNNs to combine robust pattern retrieval with uncertainty quantification.
- **Scalability:** Extend analyses to larger datasets and more complex architectures.
- **Applications:** Investigate real-world applications like noisy sensor data interpretation and memory-augmented learning.
