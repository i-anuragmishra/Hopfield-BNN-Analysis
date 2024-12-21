# Bridging Associative Memory Retrieval and Uncertainty Quantification

[![arXiv](https://img.shields.io/badge/arXiv-2412.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2412.XXXXX)

This repository contains the implementation and experimental code for the paper "Bridging Associative Memory Retrieval and Uncertainty Quantification: A Comparative Analysis of Hopfield Networks and Monte Carlo Dropout-Based Bayesian Neural Models".

## Abstract

This research provides a comparative study of classical associative memory paradigms with modern uncertainty-aware deep learning techniques. We explore the performances of Hopfield networks and Bayesian neural networks under noisy and out-of-distribution conditions. Our analysis demonstrates that Hopfield networks excel at deterministic pattern retrieval, while Bayesian neural networks provide well-calibrated uncertainty estimates.

Key findings:
- Hopfield networks achieve reliable pattern retrieval below theoretical capacity (~8 patterns)
- Bayesian neural networks exhibit well-calibrated predictive probabilities (ECE = 0.0269)
- Significant uncertainty increase (0.0094 to 0.0719) for out-of-distribution examples

## Project Structure

```
├── bayesian_nn/          # Bayesian Neural Network implementation
│   ├── model.py         # MC Dropout CNN architecture
│   └── training.py      # Training and uncertainty estimation
├── data/                # Dataset utilities and storage
├── hopfield/            # Hopfield Network implementation
│   ├── network.py      # Core Hopfield network class
│   └── utils.py        # Helper functions
└── results/            # Experimental results and visualizations
```

## Requirements

```bash
pip install -r requirements.txt
```

Dependencies:
- PyTorch ≥ 1.9.0
- NumPy ≥ 1.19.2
- Matplotlib ≥ 3.3.4
- Seaborn ≥ 0.11.1
- SciPy ≥ 1.6.2

## Key Features

1. **Hopfield Network Implementation**
   - Binary pattern storage and retrieval
   - Energy-based convergence
   - Capacity analysis
   - Noise robustness evaluation

2. **Bayesian Neural Network**
   - Monte Carlo dropout implementation
   - Uncertainty quantification
   - Out-of-distribution detection
   - Calibration assessment

## Experimental Results

The repository includes code to reproduce the main experimental results from the paper:

1. Hopfield Network Analysis
   - Pattern retrieval performance
   - Capacity limits investigation
   - Energy convergence plots
   - Pattern overlap matrices

2. Bayesian NN Evaluation
   - Uncertainty quantification
   - OOD detection performance
   - Calibration metrics
   - MNIST classification results

## Citation

If you find this work useful in your research, please cite:

```bibtex
@article{mishra2024bridging,
  title={Bridging Associative Memory Retrieval and Uncertainty Quantification: A Comparative Analysis of Hopfield Networks and Monte Carlo Dropout-Based Bayesian Neural Models},
  author={Mishra, Anurag},
  journal={arXiv preprint arXiv:2412.XXXXX},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

Anurag Mishra  
Department of Computer Science  
Rochester Institute of Technology  
am2552@rit.edu