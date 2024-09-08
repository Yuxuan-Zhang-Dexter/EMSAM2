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
sam2_checkpoint = "./sam2_hiera_large.pt"
model_cfg = "./sam2_hiera_l.yaml"
itrs = 10000

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

def resize_masks_opencv(mask, output_size=(256, 256)):
    """
    Reshapes the input NumPy array by selecting the first of the 3 redundant channels,
    then resizes each mask to the given output size using nearest-neighbor interpolation.

    Args:
    - mask (np.ndarray): Array of shape (223, 1024, 1024, 3).
    - output_size (tuple): Desired output size (H, W) for the mask. Default is (256, 256).

    Returns:
    - resized_masks: Resized masks of shape (223, 1, 256, 256).
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
    
    return resized_masks

# - Training Loop
def train_model(predictor, data, itrs, optimizer, scaler):
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
                torch.save(predictor.model.state_dict(), f"./checkpoints/all/large_model_all_{itr}.torch")
                print("Model saved at iteration:", itr + 1)

            # Accuracy (IOU) Calculation
            if itr == 0:
                mean_iou = 0
            mean_iou = mean_iou * 0.99 + 0.01 * np.mean(iou.cpu().detach().numpy())
            print("Step:", itr, "Accuracy (IOU):", mean_iou)

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
    train_model(predictor, train_data, itrs, optimizer, scaler)

if __name__ == "__main__":
    main()