# DiFrauD-detector
This project implements a robust deception detection system trained on the DiFrauD dataset, which contains over 95,000 samples across seven distinct domains. The detector uses DistilBERT fine-tuned with class-balanced training to handle diverse types of deceptive content.
Supported Domains
DomainTotal SamplesDeceptiveNon-DeceptiveClass RatioFake News20,4568,83211,6240.76xJob Scams14,29559913,6960.04xPhishing15,2726,0749,1980.66xPolitical Statements12,4978,0424,4551.80xProduct Reviews20,97110,49210,4791.00xSMS6,5741,2745,3000.24xTwitter Rumours5,7891,9693,8200.52x
Total: 95,854 samples (37,282 deceptive, 58,572 non-deceptive)
Features

Multi-domain training: Train on individual domains or all domains simultaneously
Cross-domain evaluation: Test generalization across different deception types
Class imbalance handling: Weighted sampling and loss function balancing
Early stopping: Prevents overfitting with validation-based patience
Comprehensive metrics: Accuracy, F1 (macro & binary), ROC-AUC
Automated data loading: Direct download from HuggingFace datasets

Installation
Requirements

Python 3.8 or higher
CUDA-compatible GPU (optional but recommended)
