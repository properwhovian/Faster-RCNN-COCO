## Faster R-CNN Model Training and Evaluation with COCO Dataset
This project demonstrates the process of training a Faster R-CNN model using a pre-trained ResNet-50 backbone with Feature Pyramid Networks (FPN) on the COCO dataset. It includes the following components:

## Model training using the COCO dataset.
Quantitative evaluation using Average Precision (AP).
Visualization of training metrics (loss and AP) over epochs.
## Features:
Faster R-CNN Model: A state-of-the-art object detection model, based on the ResNet-50 architecture with FPN, pre-trained on COCO.
COCO Dataset: Uses the COCO 2017 dataset for training and evaluation.
Quantitative Evaluation: Computes Average Precision (AP) for evaluating model performance.
Visualization: Plots training loss and AP over epochs for tracking model performance.
## Prerequisites
Before running the code, make sure you have the following installed:

Python 3.6 or higher
PyTorch (preferably with GPU support)
torchvision
matplotlib
fiftyone
pycocotools
To install the required libraries, you can use the following command:

```
bash
pip install torch torchvision fiftyone matplotlib pycocotools
```
## How to Run
Clone the repository:
```
bash

git clone https://github.com/properwhovian/faster-rcnn-coco.git
cd faster-rcnn-coco
```
## Download COCO dataset: 
This code uses FiftyOne's fiftyone.zoo to load the COCO dataset. The dataset will be automatically downloaded when you run the script.

## Training the Model:

Simply run the train_model.py script to start the training process. The model will use the pre-trained Faster R-CNN model and fine-tune it on the COCO dataset.
```
bash

python train_model.py
```
## Monitoring Training: The training process will log the following details:

Epoch-wise training loss.
Average Precision (AP) for the trained model.
The logs will be saved with timestamps, and they can be viewed for progress monitoring.
Visualizing Training Metrics: After the training is complete, the training loss and AP over epochs will be plotted for visualization.

The plots will display:

Training Loss: How the loss changes over each epoch.
Average Precision (AP): How the model’s object detection accuracy changes with respect to IoU threshold (0.5).
COCO Evaluation: During training, the model evaluates its performance based on the Average Precision (AP) metric. This evaluation is done at each epoch using COCOeval and the results are summarized.

## Code Overview
Model Definition: The model used is Faster R-CNN with a ResNet-50 backbone, and it is pre-trained on the COCO dataset.

Data Loading: The COCO dataset is loaded using the fiftyone library, which helps in managing large datasets and visualizing the results.

Training Loop: The training loop fine-tunes the Faster R-CNN model on the COCO dataset and calculates losses. The model is evaluated for Average Precision (AP) using COCOeval after each epoch.

Quantitative Metrics: During each epoch, the Average Precision (AP) is computed using the pycocotools library, which compares the predicted bounding boxes with the ground truth annotations.

Visualization: After training, the training loss and Average Precision (AP) are plotted using matplotlib.

## Results
After training, the model’s Average Precision (AP) and training loss will be plotted over epochs. These metrics can help evaluate the performance of the Faster R-CNN model on the COCO dataset.

## Customization
You can modify the hyperparameters such as the learning rate, batch size, and the number of epochs in the script to suit your requirements.
You can also customize the optimizer used (e.g., Adam, SGD) or modify the model architecture.

## Acknowledgments
This project is based on the Object Detection with Faster R-CNN repository by Sarojini Sharon.  
GitHub Repository: [Object Detection with Faster R-CNN](https://github.com/sarojinisharon/Object-Detection-with-Faster-R-CNN.git)
