Abstract
Neural network design still relies heavily on manually specified architectural choices, such as operator types, layer composition, width, and depth, which limits adaptability across tasks and modalities. To address this issue, we propose E-KAT (Evolutionary KolmogorovArnold-Attention-Toeplitz Network), a unified neural framework that integrates multiple structural paradigms within a single self-adaptive neuron. Specifically, E-KAT combines cross-layer attention, Toeplitz-based local operators, identity mappings, residual propagation, and edge-wise nonlinear functions inspired by the Kolmogorov–Arnold framework, and learns their relative contributions end-to-end through adaptive structural coefficients. In addition, we introduce an edge-level threshold activation mechanism with continuous soft gating and a learnable temperature parameter, enabling differentiable control over connectivity and effective information flow. These neuronal and edge structures jointly form neural clusters for single-task learning, while multiple clusters can be further coupled to support cross-task and cross-modal feature integration. 

Extensive experiments on MIT-BIH ECG, WAY-EEG-GAL, CIFAR-10, and DEAP demonstrate that E-KAT can automatically learn task-dependent structural preferences and sparse connectivity patterns. The proposed model achieves 98.63% accuracy on MIT-BIH-ECG, 96.77% on WAY-EEG-GAL, and 91.82% on CIFAR-10, while maintaining lightweight complexity, requiring only 0.54M parameters and 0.0246G MACs on ECG, 1.60M parameters and 0.1227G MACs on EEG, and 3.51M parameters and 0.10G MACs on CIFAR-10. Further analyses show that the learned connectivity patterns provide interpretable signals for effective width and depth, and that multi-cluster coupling improves robustness and feature integration in multimodal regression tasks. These results indicate that E-KAT provides an efficient and flexible alternative to manually designed architectures across heterogeneous domains. 

# E-KAT: An Interpretable Neural Framework with Adaptive Structural Composition and Edge-based Learning

> A unified neural architecture that automatically learns **how each neuron should compute** and **how information should propagate**, eliminating the need for manually designed hybrid architectures.

---

## Overview

Modern neural networks typically rely on manually designed architectural choices, such as convolution, attention, residual connections, and network depth. Although each paradigm has its own advantages, no single architecture is universally optimal across different tasks.

**E-KAT (Evolutionary Kolmogorov-Arnold Attention-Toeplitz Network)** introduces a unified adaptive neuron that integrates multiple structural paradigms into a single computational unit. Instead of manually selecting architectural components, E-KAT learns their contributions jointly during end-to-end training.

At the same time, E-KAT adopts edge-based nonlinear mappings and differentiable soft connectivity, allowing the network to automatically evolve both its internal computation and information flow according to the target task.

The framework is lightweight, interpretable, and applicable to heterogeneous data including biomedical signals and computer vision.

---

## Key Features

- **Unified Adaptive Neuron**
  - Integrates attention, convolution, identity mapping, and residual connections into a single neuron.
  - Learns the contribution of each structural component automatically.

- **Cross-layer Attention**
  - Reuses query and key representations from previous layers.
  - Reduces redundant parameters while preserving global feature modeling.

- **Toeplitz-based Local Operator**
  - Provides convolution-like local feature extraction through learnable Toeplitz matrices.

- **KAN-inspired Edge Learning**
  - Moves nonlinear activation from nodes to edges using learnable B-spline mappings.
  - Each edge learns its own nonlinear transformation.

- **Adaptive Soft Gating**
  - Learns differentiable edge connectivity through threshold-based soft masks.
  - Dynamically adjusts effective network width and information flow.

- **Interpretable Structural Evolution**
  - Learns both neuron composition and connectivity during training.
  - Provides interpretable insights into effective network depth and width.

- **Multi-cluster Learning**
  - Supports knowledge transfer across heterogeneous tasks through coupled neural clusters.

---

## Framework

E-KAT organizes computation into three hierarchical levels:

```

Task
└── Neural Cluster
└── Adaptive Neuron
└── Adaptive Edge

```

Each neuron simultaneously combines four structural operators:

```

Attention
+
Toeplitz Convolution
+
Identity Mapping
+
Residual Connection

```

Their contributions are learned automatically during optimization rather than manually specified.

---

## Adaptive Structural Composition

For each neuron, the structural operator is defined as

```

W = βW_attention + θW_toeplitz + αI

Output = WV + γx

```

where

- **β** controls global attention
- **θ** controls local convolution
- **α** controls identity propagation
- **γ** controls residual propagation

Unlike conventional hybrid networks, these coefficients are optimized directly during training, enabling neurons at different depths to specialize automatically.

---

## Adaptive Edge Learning

Instead of using fixed activation functions such as ReLU or GELU, E-KAT places nonlinear mappings on edges.

Each edge consists of

- Learnable linear mapping
- B-spline nonlinear correction
- Residual pathway
- Learnable threshold
- Differentiable soft gate

The learned edge connectivity determines how information propagates through the network while remaining fully differentiable.

---

## Experimental Results

E-KAT is evaluated on heterogeneous datasets spanning biomedical signal analysis and computer vision.

| Dataset | Task | Accuracy |
|----------|-----------------------|----------|
| MIT-BIH ECG | ECG Classification | **98.63%** |
| WAY-EEG-GAL | EEG Detection | **96.77%** |
| CIFAR-10 | Image Classification | **91.82%** |

The proposed framework achieves competitive accuracy while maintaining a lightweight model size and low computational complexity.

---

## Repository Structure

```

E-KAT/
│
├── datasets/            # Dataset preparation
├── models/              # Network implementation
├── layers/              # Adaptive neurons and edge modules
├── training/            # Training scripts
├── evaluation/          # Evaluation scripts
├── visualization/       # Structural evolution visualization
├── checkpoints/         # Pretrained models
├── utils/               # Utility functions
└── README.md

```

---

## Main Contributions

- Introduces a unified adaptive neuron capable of learning structural composition during training.
- Proposes a cross-layer attention mechanism with reduced parameter redundancy.
- Develops adaptive edge learning based on KAN-inspired nonlinear mappings.
- Introduces differentiable soft connectivity for adaptive information routing.
- Enables interpretable structural evolution at both neuron and network levels.
- Extends the framework to multi-task learning through coupled neural clusters.

---

## Citation

If you find this work useful, please cite

```bibtex
@article{ding2026ekat,
  title={E-KAT: An Interpretable Neural Framework with Adaptive Structural Composition and Edge-based Learning},
  author={Yunhe Ding},
  journal={Preprint},
  year={2026}
}
```

---

## License

This project is released under the MIT License.

---

## Acknowledgements

This work was developed at the Department of Electrical Engineering and Computer Science, University of California, Irvine.
