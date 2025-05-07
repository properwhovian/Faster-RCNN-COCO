 Object Detection 

## Overview

This project aims to develop an object detection pipeline using the COCO dataset and implements the following tasks:

- **Data Distillation**: A method for reducing the dataset size without sacrificing too much accuracy.
- **Knowledge Distillation**: A student model is trained to learn from a teacher model using knowledge transfer techniques.

## Repository Structure
```bash
/Object Detection/
│
├── data_processing/
│   ├── data_preprocessing.py  # Code for importing, cleaning, and processing dataset
│   ├── data_augmentation.py   # Data augmentation techniques (if applicable)
│   ├── create_splits.py       # Code for creating training, validation, and testing splits
│   └── README.md              # Documentation for data processing
│
├── base_model/
│   ├── model.py               # Full implementation of the main model architecture
│   ├── train.py               # Training script for the model
│   ├── evaluate.py            # Evaluation script for the model
│   ├── checkpoints/           # Directory for saved model checkpoints or weights
│   ├── report_1_metrics.py    # Code to reproduce performance metrics from Report 1
│   └── README.md              # Documentation for base model
│
├── data_distillation/
│   ├── distillation.py        # Implementation of data distillation approach
│   ├── generate_synthetic_samples.py  # Code for generating synthetic samples
│   ├── distillation_evaluation.py    # Scripts measuring performance trade-offs
│   ├── performance_comparison.py     # Code comparing performance loss and training time
│   ├── distillation_results/        # Directory for results from Report 2
│   └── README.md              # Documentation for data distillation
│
├── knowledge_distillation/
│   ├── student_model.py       # Student model architecture implementation
│   ├── train_student.py       # Complete training pipeline for the student model
│   ├── teacher_model.py       # Teacher model implementation
│   ├── distillation_scheme.py # Implementation of knowledge distillation method/scheme
│   ├── evaluation.py          # Evaluation scripts producing results identical to Report 3
│   └── README.md              # Documentation for knowledge distillation
│
├── environment/
│   ├── environment.yml       # Conda environment file with exact library versions
│   ├── setup_instructions.md # Setup instructions for the environment
│   └── system_requirements.md # Additional dependencies or system requirements
│
├── README.md                 # Comprehensive README for the entire repository


