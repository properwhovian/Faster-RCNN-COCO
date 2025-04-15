# === Load the Teacher Model for Inference Only ===
teacher_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
teacher_model.to(device)
teacher_model.eval()

# === Generate Pseudo-labels using Teacher Model ===
pseudo_labels = []

logger.info("Generating pseudo-labels with the teacher model...")

for sample in train_dataset.take(200):  # Limit for quicker experimentation
    image_path = sample.filepath
    image = Image.open(image_path).convert("RGB")
    image_tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = teacher_model(image_tensor)[0]

    # Threshold scores
    threshold = 0.7
    boxes = output["boxes"][output["scores"] > threshold].cpu()
    labels = output["labels"][output["scores"] > threshold].cpu()

    if len(boxes) == 0:
        continue

    pseudo_labels.append({
        "image_path": image_path,
        "boxes": boxes,
        "labels": labels
    })

logger.info("Pseudo-label generation complete. Starting student training...")

# === Define a Lighter Student Model ===
student_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)  # Not pretrained
student_model.to(device)
student_model.train()

# === Dataset for Student ===
class PseudoLabelDataset(torch.utils.data.Dataset):
    def __init__(self, pseudo_labels):
        self.data = pseudo_labels
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        image = self.transform(image)
        target = {
            "boxes": item["boxes"].float(),
            "labels": item["labels"],
            "image_id": torch.tensor([idx]),
            "area": (item["boxes"][:, 2] - item["boxes"][:, 0]) *
                    (item["boxes"][:, 3] - item["boxes"][:, 1]),
            "iscrowd": torch.zeros(len(item["labels"]), dtype=torch.int64)
        }
        return image, target

    def __len__(self):
        return len(self.data)

pseudo_loader = DataLoader(PseudoLabelDataset(pseudo_labels), batch_size=batch_size, shuffle=True)

# === Optimizer for Student ===
student_optimizer = optim.Adam(student_model.parameters(), lr=learning_rate)

# === Train Student Model ===
train_model(student_model, pseudo_loader, student_optimizer, num_epochs=num_epochs)
plot_metrics(train_losses, average_precisions)
