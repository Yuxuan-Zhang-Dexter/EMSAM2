import os
import torch
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torchvision.ops import masks_to_boxes
from transformers import (
    AdamW,
    AutoModelForCausalLM,
    AutoProcessor,
    get_scheduler
)
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

from torch.amp import autocast

# Paths for image data
data_dir = Path("./snemi/")
raw_image_dir = data_dir / 'image_pngs'
seg_image_dir = data_dir / 'seg_pngs'

# Global Constants
CHECKPOINT = "microsoft/Florence-2-large-ft"
REVISION = 'refs/pr/19'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available

BATCH_SIZE = 6
NUM_WORKERS = 0
label_num = 170  # Limit number of labels to 170
EPOCHS = 10000
LR = 5e-6
rank = 8
alpha = 8

# Load the model and processor
model = AutoModelForCausalLM.from_pretrained(CHECKPOINT, trust_remote_code=True, revision=REVISION)
processor = AutoProcessor.from_pretrained(CHECKPOINT, trust_remote_code=True, revision=REVISION)

model.to(DEVICE)  # Move model to the selected device

# Prepare dataset (images and corresponding segmentations)
data = []
for ff, name in enumerate(os.listdir(raw_image_dir)):
    data.append({
        'image': raw_image_dir / f'image{ff:04d}.png',
        'annotation': seg_image_dir / f'seg{ff:04d}.png'
    })

# Split dataset into training and validation sets
valid_data = data[80:]  # 20% validation set
data = data[:80]  # 80% training set

# Convert segmentation masks into bounding boxes (object detection task)
def convert_mask2box(mask: np.ndarray):
    inds = np.unique(mask)[1:]  # Get unique object indices (excluding background)
    masks = [mask == ind for ind in inds]
    masks_tensor = torch.from_numpy(np.array(masks))  # Convert masks to tensor
    boxes = masks_to_boxes(masks_tensor)  # Convert masks to bounding boxes
    return boxes.numpy()

# Normalize bounding box coordinates (to fit within a predefined resolution)
def normalize_loc(prefix: str, instance_type: str, image_path: str, mask: np.ndarray, input_boxes: np.ndarray):
    x_res, y_res = mask.shape
    normal_boxes = [[box[0] / x_res * 1000, box[1] / y_res * 1000, box[2] / x_res * 1000, box[3] / y_res * 1000]
                    for box in input_boxes]  # Normalize boxes
    normal_boxes = np.rint(normal_boxes)  # Round to nearest integer
    suffix = ''
    count = 0
    for i in range(len(normal_boxes)):
        if count == label_num:  # Stop when reaching max label count
            break
        x1, y1, x2, y2 = map(int, normal_boxes[i])
        suffix += f"{instance_type}<loc_{x1}><loc_{y1}><loc_{x2}><loc_{y2}>"
        count += 1
    return {"image": image_path, "prefix": prefix, "suffix": suffix}

# Process dataset by converting masks to bounding boxes and normalizing them
def prepare_dataset(data, instance_type, prefix):
    dataset = []
    for element in data:
        image_path = element['image']
        seg_path = element['annotation']
        mask = np.array(Image.open(seg_path))  # Load segmentation mask
        input_boxes = convert_mask2box(mask)  # Convert mask to bounding boxes
        curated_data = normalize_loc(prefix, instance_type, image_path, mask, input_boxes)
        dataset.append(curated_data)
    return dataset

# Prepare training and validation datasets
train_dataset = prepare_dataset(data, 'neuron', "<OD>")
val_dataset = prepare_dataset(valid_data, 'neuron', "<OD>")

# Dataset class for detection tasks
class DetectionDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        image = cv2.imread(str(data['image']))  # Read image
        prefix = data['prefix']
        suffix = data['suffix']
        return prefix, suffix, image

# Collate function for DataLoader
def collate_fn(batch):
    questions, answers, images = zip(*batch)
    inputs = processor(text=list(questions), images=list(images), return_tensors="pt", padding=True).to(DEVICE)
    return inputs, answers

# Initialize DataLoader for training and validation
train_loader = DataLoader(DetectionDataset(train_dataset), batch_size=BATCH_SIZE,
                          collate_fn=collate_fn, num_workers=NUM_WORKERS, shuffle=True)
val_loader = DataLoader(DetectionDataset(val_dataset), batch_size=BATCH_SIZE,
                        collate_fn=collate_fn, num_workers=NUM_WORKERS, shuffle=False)

# LoRA configuration for model fine-tuning
TARGET_MODULES = ["q_proj", "o_proj", "k_proj", "v_proj", "linear", "Conv2d", "lm_head", "fc2"]

# LoRA setup for parameter-efficient fine-tuning
config = LoraConfig(
    r=rank,
    lora_alpha=alpha,
    target_modules=TARGET_MODULES,
    task_type="CAUSAL_LM",
    lora_dropout=0.05,
    bias="none",
    inference_mode=False,
    use_rslora=True,
    init_lora_weights="gaussian",
    revision=REVISION
)

# Apply LoRA to the model
peft_model = get_peft_model(model, config)
peft_model.print_trainable_parameters()


# Use DataParallel for multi-GPU training
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    peft_model = torch.nn.DataParallel(peft_model)

# Training loop for fine-tuning the model
def train_model(train_loader, val_loader, model, processor, epochs=10, lr=1e-6):
    optimizer = AdamW(model.parameters(), lr=lr)
    num_training_steps = epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for inputs, answers in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}"):
            input_ids = inputs["input_ids"]
            pixel_values = inputs["pixel_values"]
            labels = processor.tokenizer(text=answers, return_tensors="pt", padding=True).input_ids.to(DEVICE)

            with autocast('cuda'):
                outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
                loss = outputs.loss.mean()

            loss.backward()  # Compute gradients
            optimizer.step()  # Update weights
            lr_scheduler.step()  # Update learning rate
            optimizer.zero_grad()  # Reset gradients
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        print(f"Average Training Loss: {avg_train_loss}")

        # Validation phase (no gradients needed)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, answers in tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}/{epochs}"):
                input_ids = inputs["input_ids"]
                pixel_values = inputs["pixel_values"]
                labels = processor.tokenizer(text=answers, return_tensors="pt", padding=True).input_ids.to(DEVICE)
                with autocast('cuda'):
                    outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
                    loss = outputs.loss.mean()
                val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            print(f"Average Validation Loss: {avg_val_loss}")

        # Save model checkpoint every 100 epochs
        if (epoch + 1) % 100 == 0:
            output_dir = f"./model_checkpoints/large_model/epoch_{epoch+1}"
            os.makedirs(output_dir, exist_ok=True)
            model.module.save_pretrained(output_dir)  # Save model
            processor.save_pretrained(output_dir)  # Save processor

# Start the training process
train_model(train_loader, val_loader, peft_model, processor, epochs=EPOCHS, lr=LR)