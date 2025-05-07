# Final Project: Object Detection with Knowledge Distillation

## Overview

This project aims to develop an object detection pipeline using the COCO dataset and implements the following tasks:

- **Data Distillation**: A method for reducing the dataset size without sacrificing too much accuracy.
- **Knowledge Distillation**: A student model is trained to learn from a teacher model using knowledge transfer techniques.

## Repository Structure

```bash
/final_project/
│
├── /data/                        # Raw data or scripts to download and preprocess data
│   ├── /raw/                     # Raw dataset ( COCO dataset)
│   ├── /processed/                # Processed datasets (after cleaning, augmentation, etc.)
│   ├── download_data.py           # Script to download and preprocess data
│
├── /models/                       # Model implementations
│   ├── base_model.py              # Base model architecture (Report 1)
│   ├── student_model.py           # Student model architecture (Report 3)
│   ├── train.py                   # Training script for the model
│   ├── evaluate.py                # Evaluation script to reproduce metrics from Report 1
│   ├── distillation.py            # Data distillation implementation (Report 2)
│   ├── distillation_analysis.py   # Code for generating synthetic samples and performance trade-offs
│
├── /notebooks/                    # Jupyter Notebooks 
│   ├── data_processing.ipynb      # Data processing and analysis notebook
│   ├── model_training.ipynb       # Training and evaluation notebook for the base model
│   ├── distillation.ipynb         # Notebook showing the performance comparison and analysis
│
├── /scripts/                      #  scripts for environment setup or utilities
│   ├── environment.yml            # Conda environment configuration
│   ├── requirements.txt           # Python dependencies
│   ├── setup.py                   # Script for installation setup
│
├── /results/                      # Folder to save model checkpoints, logs, and evaluation results
│   ├── model_checkpoints/         # Saved models
│   ├── logs/                      # Training logs
│
├── .gitignore                     # Git ignore file to exclude unnecessary files
└── README.md                      # Comprehensive project documentation

