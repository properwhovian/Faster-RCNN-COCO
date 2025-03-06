import logging
import torch
import torchvision
import fiftyone as fo
import fiftyone.zoo as foz
from torchvision.models.detection import FasterRCNN
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import nn, optim
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger.info("All necessary libraries loaded successfully.")

# Hyperparameters
learning_rate = 1e-4
batch_size = 4
num_epochs = 5
optimizer_type = "Adam"

# Log hyperparameters
logger.info(f"Training Hyperparameters:")
logger.info(f"Learning Rate: {learning_rate}")
logger.info(f"Batch Size: {batch_size}")
logger.info(f"Number of Epochs: {num_epochs}")
logger.info(f"Optimizer: {optimizer_type}")

# Check device (GPU or CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Load COCO dataset
try:
    train_dataset = foz.load_zoo_dataset("coco-2017", split="train", dataset_name="coco-2017-train")
    logger.info("COCO 2017 training dataset loaded successfully.")
except Exception as e:
    logger.error(f"Error loading dataset: {e}")
    raise e

# Load the pre-trained Faster R-CNN model
try:
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.to(device)
    model.eval()
    logger.info("Faster R-CNN model loaded and set to evaluation mode.")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise e

# Create COCO ground truth and predictions list
ground_truths = []
predictions = []

# Lists to store metrics for plotting
train_losses = []
average_precisions = []

# Define the training loop
def train_model(model, train_loader, optimizer, num_epochs=5):
    model.train()  # Set the model to training mode

    for epoch in range(num_epochs):
        running_loss = 0.0
        for step, (images, targets) in enumerate(train_loader, start=1):
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            losses.backward()
            optimizer.step()

            running_loss += losses.item()

            # Log training progress
            if step % 10 == 0:
                logger.info(f"Epoch: [{epoch+1}/{num_epochs}], Step: [{step}/{len(train_loader)}], "
                            f"Training Loss: {losses.item():.4f}")

            # Collect ground truth and predictions for evaluation
            for target, prediction in zip(targets, model(images)):
                boxes = prediction['boxes'].cpu().detach().numpy()
                scores = prediction['scores'].cpu().detach().numpy()
                labels = prediction['labels'].cpu().detach().numpy()

                # Prepare predictions for COCO evaluation
                for i in range(len(boxes)):
                    ground_truths.append({
                        "image_id": target['image_id'].item(),
                        "category_id": labels[i],
                        "bbox": boxes[i],
                        "area": (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]),
                        "iscrowd": 0,
                        "id": len(ground_truths) + 1
                    })
                    predictions.append({
                        "image_id": target['image_id'].item(),
                        "category_id": labels[i],
                        "bbox": boxes[i],
                        "score": scores[i]
                    })

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        logger.info(f"Epoch: [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss:.4f}")

        # Compute Average Precision (AP) after each epoch
        ap = compute_ap(ground_truths, predictions)
        average_precisions.append(ap)

# Function to compute Average Precision (AP)
def compute_ap(ground_truths, predictions):
    # Create a COCO object for ground truth and predictions
    coco_gt = COCO()
    coco_gt.dataset['images'] = [{"id": i, "file_name": f"image_{i}.jpg", "width": 640, "height": 480} for i in range(1, len(ground_truths) + 1)]  # Define image metadata
    coco_gt.dataset['annotations'] = ground_truths
    coco_gt.dataset['categories'] = [{'id': i, 'name': str(i)} for i in range(1, 91)]  # COCO has 80 classes
    coco_gt.createIndex()

    coco_dt = coco_gt.loadRes(predictions)

    # Initialize COCOeval object
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')

    # Evaluate the predictions
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Extract Average Precision (AP) for IoU threshold 0.5 (AP50)
    ap = coco_eval.stats[0]  # AP at IoU=0.5
    return ap

# DataLoader for training data (Updated code to use FiftyOne's sample access method)
class CustomFiftyOneDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.transform = transforms.Compose([transforms.ToTensor()])  # Example transform
    
    def __getitem__(self, idx):
        # Get the sample ID
        sample = self.dataset[idx]
        
        # Access the image and annotation from the sample
        image_path = sample['filepath']
        annotations = sample['ground_truth']
        
        # Process the image (open, transform, etc.)
        image = Image.open(image_path)
        image = self.transform(image)
        
        # Prepare the targets for the model
        targets = []
        for annotation in annotations.detections:
            target = {
                'bbox': annotation.bounding_box,  # Assuming it's in [xmin, ymin, xmax, ymax]
                'category_id': annotation.label,
                'area': annotation.area,
                'iscrowd': 0,
                'image_id': sample.id
            }
            targets.append(target)

        return image, targets

# DataLoader for training data
train_loader = DataLoader(CustomFiftyOneDataset(train_dataset), batch_size=batch_size, shuffle=True)

# Initialize the optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Start the training process
train_model(model, train_loader, optimizer, num_epochs=num_epochs)

# Plot the metrics (Training Loss and Average Precision)
def plot_metrics(train_losses, average_precisions):
    epochs = range(1, len(train_losses) + 1)
    
    # Plot Training Loss
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss', color='b')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.grid(True)
    plt.legend()

    # Plot Average Precision (AP)
    plt.subplot(1, 2, 2)
    plt.plot(epochs, average_precisions, label='Average Precision (AP)', color='g')
    plt.xlabel('Epochs')
    plt.ylabel('Average Precision (AP)')
    plt.title('Average Precision (AP) Over Epochs')
    plt.grid(True)
    plt.legend()

    # Show the plots
    plt.tight_layout()
    plt.show()

# After training, plot the metrics
plot_metrics(train_losses, average_precisions)


# Launching FiftyOne session
#session = fo.launch_app()
#session.dataset = train_dataset
