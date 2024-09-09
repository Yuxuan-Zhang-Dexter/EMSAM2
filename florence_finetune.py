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

import albumentations as A

# Paths for image data
data_dir = Path("./snemi/")
raw_image_dir = data_dir / 'image_pngs'
seg_image_dir = data_dir / 'seg_pngs'
raw_image_slice_dir = data_dir / 'image_slice_pngs'
seg_image_slice_dir = data_dir / 'seg_slice_pngs'
log_dir="./logs"
os.makedirs(raw_image_slice_dir, exist_ok=True)
os.makedirs(seg_image_slice_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

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
weight_decay=1e-2

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


# Slice Image and Masks for sequence length limit
# - slice image and segmentation for florence sequence length limit
def create_slices(image_path, slice_image_dir):
    img = Image.open(image_path)
     # Get image dimensions
    width, height = img.size
    
    # Calculate the midpoint
    mid_x, mid_y = width // 2, height // 2
    
    # Define the four slices (left, upper, right, lower)
    slices = {
        'top_left': (0, 0, mid_x, mid_y),
        'top_right': (mid_x, 0, width, mid_y),
        'bottom_left': (0, mid_y, mid_x, height),
        'bottom_right': (mid_x, mid_y, width, height)
    }
    
    # Loop through the slices, crop, and save them
    all_slices = []
    for key, coords in slices.items():
        slice_img = img.crop(coords)
        # Format the name: base name + coordinates
        slice_filename = f"{image_path.stem}_{coords[0]}_{coords[1]}_{coords[2]}_{coords[3]}.png"
        slice_img.save( slice_image_dir / slice_filename)
        all_slices.append( slice_image_dir / slice_filename)

    return all_slices

def slice_all_image_seg(data, raw_image_slice_dir, seg_image_slice_dir):
    new_data = []
    for element in data:
        image_path = element['image']
        seg_path = element['annotation']
        image_lst = create_slices(image_path, raw_image_slice_dir)
        seg_lst = create_slices(seg_path, seg_image_slice_dir)

        for i in range(len(image_lst)):
            new_data.append({'image': image_lst[i], 'annotation': seg_lst[i]})
        
    return new_data

data = slice_all_image_seg(data, raw_image_slice_dir, seg_image_slice_dir)
valid_data = slice_all_image_seg(valid_data, raw_image_slice_dir, seg_image_slice_dir)


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

# - Apply Albumentations to image and bounding boxes
# Albumentations transformation pipeline
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(scale_limit=0.1, rotate_limit=45, shift_limit=0.1, p=0.5),
    A.RandomBrightnessContrast(p=0.5)
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

# - Check Invalid Bounding Boxes
def validate_boxes(boxes):
    """Validate and filter out invalid bounding boxes."""
    valid_boxes = []
    for box in boxes:
        x_min, y_min, x_max, y_max = box[:4]
        if x_max > x_min and y_max > y_min:
            valid_boxes.append(box)
    return valid_boxes



def apply_augmentation(image, input_boxes):

    class_labels = [0] * len(input_boxes)  # All neurons are of the same class (update if multi-class)
    augmented = transform(image=image, bboxes=input_boxes, class_labels=class_labels)
    
    return augmented['image'], augmented['bboxes']

def save_augmented_image(image, original_image_path, augmented_image_dir):
    # Convert image array back to image format
    augmented_image = Image.fromarray(image)
    
    # Generate a new file name based on the original
    new_image_filename = f"{original_image_path.stem}_augmented.png"
    new_image_path = augmented_image_dir / new_image_filename
    
    # Save the new image
    augmented_image.save(new_image_path)
    
    return new_image_path

## - Prepare all training dataset and validation dataset
def prepare_dataset(data, instance_type, prefix, augmentation = False):
    dataset = []
    for element in data:
        image_path = element['image']
        seg_path = element['annotation']
        mask = np.array(Image.open(seg_path))
        input_boxes = convert_mask2box(mask)
        # Validate bounding boxes before applying augmentation
        input_boxes = validate_boxes(input_boxes)
        # - Start data augmentation
        if augmentation == True:
            temp_image = np.array(Image.open(image_path))
            augmented_image, augmented_boxes = apply_augmentation(temp_image, input_boxes)
            new_image_path = save_augmented_image(augmented_image, image_path, image_path.parent)
            new_curated_data = normalize_loc(prefix, instance_type, new_image_path, mask, augmented_boxes)
            dataset.append(new_curated_data)

        curated_data = normalize_loc(prefix, instance_type, image_path, mask, input_boxes)
        dataset.append(curated_data)
    print(f"Remove Invalid Bounding Boxes xmin >= xlarge or ymin >= ylarge")
    return dataset

train_dataset = prepare_dataset(data, 'neuron', "<OD>", True)
val_dataset = prepare_dataset(valid_data, 'neuron', "<OD>", True)

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
def train_model(train_loader, val_loader, model, processor, epochs=10, lr=1e-6, weight_decay=1e-2):
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    num_training_steps = epochs * len(train_loader)
    # Total number of training steps
    num_training_steps = epochs * len(train_loader)
    
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),  # 10% warmup steps
        num_training_steps=num_training_steps,)

    # Open log files for saving training and validation loss
    train_loss_file = open(os.path.join(log_dir, "flo_train_loss.txt"), "w")
    val_loss_file = open(os.path.join(log_dir, "flo_val_loss.txt"), "w")

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

        # Save training loss for this epoch
        train_loss_file.write(f"{epoch + 1},{avg_train_loss}\n")
        train_loss_file.flush()

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

            # Save validation loss for this epoch
            val_loss_file.write(f"{epoch + 1},{avg_val_loss}\n")
            val_loss_file.flush()
        
        if (epoch + 1) < 1000:
            if (epoch + 1) % 100 == 0:
                output_dir = f"./model_checkpoints/large_model/epoch_{epoch+1}"
                os.makedirs(output_dir, exist_ok=True)
                model.module.save_pretrained(output_dir)  # Save model
                processor.save_pretrained(output_dir) 

        # Save model checkpoint every 100 epochs
        if (epoch + 1) % 1000 == 0:
            output_dir = f"./model_checkpoints/large_model/epoch_{epoch+1}"
            os.makedirs(output_dir, exist_ok=True)
            model.module.save_pretrained(output_dir)  # Save model
            processor.save_pretrained(output_dir)  # Save processor
    # Close the log files
    train_loss_file.close()
    val_loss_file.close()

# Start the training process
train_model(train_loader, val_loader, peft_model, processor, epochs=EPOCHS, lr=LR, weight_decay=weight_decay)