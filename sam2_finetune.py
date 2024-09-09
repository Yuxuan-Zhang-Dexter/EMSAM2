import numpy as np
import torch
import cv2
import os
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from torchvision.ops import masks_to_boxes
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Set up environment variables
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# - Global Variables
data_dir = Path("./snemi/")
raw_image_dir = data_dir / 'image_pngs'
seg_image_dir = data_dir / 'seg_pngs'
train_loss_filename = "sam_train_loss_full.txt"
val_loss_filename = "sam_val_loss_full.txt"
log_dir="./logs"
sam2_checkpoint = "./sam2_hiera_large.pt"
model_cfg = "./sam2_hiera_l.yaml"
itrs = 10000
val_num = 10

# Create checkpoint directory if it doesn't exist
checkpoint_dir = './checkpoints/all'
os.makedirs(checkpoint_dir, exist_ok=True)

DEVICE = 'cpu'

# - Prepare Dataset
def load_data():
    data = []
    for ff, name in enumerate(os.listdir(raw_image_dir)):
        data.append({
            'image': raw_image_dir / f'image{ff:04d}.png',
            'annotation': seg_image_dir / f'seg{ff:04d}.png'
        })
    return data[:80], data[80:]  # return train_data, valid_data

# Function to read and process a batch of data
def read_batch(data):
    ent = data[np.random.randint(len(data))]  # choose random entry
    Img = cv2.imread(str(ent["image"]))  # read image
    ann_map_grayscale = np.array(Image.open(ent['annotation']))
    ann_map = np.stack((ann_map_grayscale, ) * 3, axis=-1)
   
    # resize image
    r = np.min([1024 / Img.shape[1], 1024 / Img.shape[0]])  # scaling factor
    Img = cv2.resize(Img, (int(Img.shape[1] * r), int(Img.shape[0] * r)))
    ann_map = cv2.resize(ann_map, (int(ann_map.shape[1] * r), int(ann_map.shape[0] * r)), interpolation=cv2.INTER_NEAREST)

    # get bounding boxes from the mask
    inds = np.unique(ann_map_grayscale)[1:]  # load all indices
    masks = [(ann_map_grayscale == ind) for ind in inds]
    masks_tensor = torch.from_numpy(np.array(masks))
    boxes = masks_to_boxes(masks_tensor)
    input_boxes = boxes.numpy()

    # Get binary masks and points
    points = []
    masks = []
    for ind in inds:
        mask = (ann_map == ind).astype(np.uint8)
        masks.append(mask)
        coords = np.argwhere(mask > 0)  # get all coordinates in mask
        yx = np.array(coords[np.random.randint(len(coords))])  # choose random point
        points.append([[yx[1], yx[0]]])
   
    return Img, np.array(masks), np.array(points), input_boxes, np.ones([len(masks), 1])

# - Build SAM2 Model
def build_sam2_model():
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=DEVICE)
    return SAM2ImagePredictor(sam2_model)

# - Set Training Parameters
def set_training_mode(predictor):
    predictor.model.sam_mask_decoder.train(True)
    predictor.model.sam_prompt_encoder.train(True)
    predictor.model.image_encoder.train(True)

def mask_to_logits(mask, epsilon=1e-6):
    """
    Convert binary mask to mask logits.

    Args:
        mask (torch.Tensor or np.ndarray): A binary mask tensor with values in {0, 1}.
        epsilon (float): A small value to prevent division by zero.
    
    Returns:
        torch.Tensor: The corresponding logits.
    """
    # Ensure the mask is in float32 and has values in range [0, 1]
    mask = mask.astype(np.float32)
    
    # Apply the logit function: log(p / (1 - p))
    logits = np.log(mask + epsilon) - np.log(1 - mask + epsilon)
    
    return logits


def resize_masks_opencv(mask, output_size=(256, 256)):
    """
    Reshapes the input NumPy array by selecting the first of the 3 redundant channels,
    then resizes each mask to the given output size using nearest-neighbor interpolation.
    After resizing, the binary masks (0 and 1) are converted to logits.

    Args:
    - mask (np.ndarray): Array of shape (223, 1024, 1024, 3).
    - output_size (tuple): Desired output size (H, W) for the mask. Default is (256, 256).

    Returns:
    - resized_mask_logits: Resized mask logits of shape (223, 1, 256, 256).
    """
    # Select the first channel (shape becomes (223, 1024, 1024))
    masks_single_channel = mask[..., 0]
    
    # Reshape to (223, 1, 1024, 1024) to add the single channel back using np.expand_dims
    reshaped_masks = np.expand_dims(masks_single_channel, axis=1)

    # Initialize the array to store resized masks (223, 1, 256, 256)
    resized_masks = np.zeros((reshaped_masks.shape[0], 1, output_size[0], output_size[1]), dtype=reshaped_masks.dtype)
    
    # Loop over each mask and resize using cv2.resize with nearest-neighbor interpolation
    for i in range(reshaped_masks.shape[0]):
        resized_masks[i, 0, :, :] = cv2.resize(reshaped_masks[i, 0, :, :], output_size, interpolation=cv2.INTER_NEAREST)

    # Convert the resized binary masks (0 and 1) to mask logits
    resized_mask_logits = np.zeros_like(resized_masks, dtype=np.float32)
    for i in range(resized_masks.shape[0]):
        resized_mask_logits[i, 0, :, :] = mask_to_logits(resized_masks[i, 0, :, :])
    
    return resized_mask_logits

# - Training Loop
def train_model(predictor, data, valid_data, itrs, optimizer, scaler):
    train_loss_file = open(os.path.join(log_dir, train_loss_filename), "w")
    val_loss_file = open(os.path.join(log_dir, val_loss_filename), "w")
    for itr in range(itrs):
        with torch.amp.autocast(device_type=DEVICE):
            image, mask, input_point, input_boxes, input_label = read_batch(data)
            if mask.shape[0] == 0:
                continue
            predictor.set_image(image)

            reshaped_masks = resize_masks_opencv(mask)

            # - prompt encoding
            mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(input_point, input_label, box=input_boxes, mask_logits=reshaped_masks, normalize_coords=True)
            sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(points=(unnorm_coords, labels), boxes=unnorm_box, masks=mask_input)

            # - mask decoder
            batched_mode = unnorm_coords.shape[0] > 1
            high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
            low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
                image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
                image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
                repeat_image=batched_mode,
                high_res_features=[feat for feat in high_res_features],
            )

            prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])

            # - segmentation loss calculation
            gt_mask = torch.tensor(mask.astype(np.float32))[:, :, :, 0]
            prd_mask = torch.sigmoid(prd_masks[:, 0])
            seg_loss = (-gt_mask * torch.log(prd_mask + 0.00001) - (1 - gt_mask) * torch.log((1 - prd_mask) + 0.00001)).mean()

            # - score loss calculation (IOU)
            inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
            iou = inter / (gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter)
            score_loss = torch.abs(prd_scores[:, 0] - iou).mean()

            # - backpropagation
            loss = (seg_loss + score_loss * 0.05)
            predictor.model.zero_grad()  # empty gradient
            scaler.scale(loss).backward()  # Backpropagate
            scaler.step(optimizer)
            scaler.update()  # Mix precision

            # Save model every 1000 iterations
            if (itr + 1) % 1000 == 0:
                torch.save(predictor.model.state_dict(), f"./checkpoints/all/large_model_full_{itr + 1}.torch")
                print("Model saved at iteration:", itr + 1)

            # Accuracy (IOU) Calculation
            if itr == 0:
                mean_iou = 0
            mean_iou = mean_iou * 0.99 + 0.01 * np.mean(iou.cpu().detach().numpy())
            print(f"Training IOU at iteration {itr + 1}: {mean_iou}")
            # Save training loss for this epoch
            train_loss_file.write(f"{itr + 1},{mean_iou}\n")
            train_loss_file.flush()
        # - Evaluation Step
        # Evaluation step on validation data after each iteration

        total_iou = 0
        for i in range(val_num):
            with torch.no_grad():  # Disable gradient calculation for inference
                img, mask, input_points, input_boxes, input_labels = read_batch(valid_data)

                predictor.set_image(img)  # Set image in the predictor (Image Encoder)

                # Prompt Encoder + Mask Decoder
                masks, scores, logits = predictor.predict(
                    point_coords=input_points,
                    point_labels=input_labels,
                    box=input_boxes,
                    multimask_output=False
                )

                prd_mask = torch.sigmoid(torch.tensor(masks[:, 0], dtype=torch.float32))
                gt_mask = torch.tensor(mask.astype(np.float32))[:, :, :, 0]

                # Calculate IOU for validation data
                inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
                iou_val = inter / (gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter)

                total_iou += iou_val.mean().cpu().numpy()
    
        avg_iou = total_iou / val_num
        print(f"Validation IOU at iteration {itr}: {avg_iou}")

        # Save validation loss for this epoch
        val_loss_file.write(f"{itr + 1},{avg_iou}\n")
        val_loss_file.flush()
    # Close the log files
    train_loss_file.close()
    val_loss_file.close()

# - Main function to run the script
def main():
    train_data, valid_data = load_data()  # Load data

    # Build the model
    predictor = build_sam2_model()

    # Set training mode
    set_training_mode(predictor)

    # Define optimizer and scaler for mixed precision
    optimizer = torch.optim.AdamW(params=predictor.model.parameters(), lr=1e-5, weight_decay=4e-5)
    scaler = torch.amp.GradScaler()

    # Start training
    train_model(predictor, train_data, valid_data, itrs, optimizer, scaler)

if __name__ == "__main__":
    main()