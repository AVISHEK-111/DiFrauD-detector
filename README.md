# DiFrauD-detector
This project implements a robust deception detection system trained on the DiFrauD dataset, which contains over 95,000 samples across seven distinct domains. The detector uses DistilBERT fine-tuned with class-balanced training to handle diverse types of deceptive content.

Supported Domains: 
Fake News, Job Scams, Phishing, Political Statements, Product Reviews, Twitter Rumours.

Total: 95,854 samples (37,282 deceptive, 58,572 non-deceptive)

Features:

Multi-domain training: Train on individual domains or all domains simultaneously
Cross-domain evaluation: Test generalization across different deception types
Class imbalance handling: Weighted sampling and loss function balancing
Early stopping: Prevents overfitting with validation-based patience
Comprehensive metrics: Accuracy, F1 (macro & binary), ROC-AUC
Automated data loading: Direct download from HuggingFace datasets

Installation Requirements

Python 3.8 or higher

CUDA-compatible GPU (optional but recommended)
