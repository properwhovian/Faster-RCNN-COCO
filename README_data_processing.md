# Data Processing for COCO Dataset

This folder contains all the necessary components for processing the COCO dataset.

## Files:

### `data_preprocessing.py`
- Contains functions for loading COCO annotations, extracting category names, and loading images.

### `data_augmentation.py`
- Contains the `CustomTransform` class for applying data augmentation techniques such as random horizontal flips, rotations, and color jittering.

### `create_splits.py`
- Contains a function for splitting the dataset into training, validation, and test sets.

## Instructions:
1. **Loading Annotations**: Use `data_preprocessing.py` to load annotations and category names.
2. **Augmenting Data**: Use `data_augmentation.py` to apply transformations to the images.
3. **Splitting Dataset**: Use `create_splits.py` to split the dataset into training, validation, and test sets.

You can run each module independently or combine them as needed for your data processing pipeline.
