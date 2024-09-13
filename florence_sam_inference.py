# File: florence_sam_inference.py

# - System and OS packages
import os
import sys
from pathlib import Path

# - Numerical and scientific libraries
import numpy as np
import random

# - PyTorch and related libraries
import torch
from torch.utils.data import Dataset, DataLoader

# - Computer vision and image processing
import cv2
from PIL import Image
import torchvision.transforms.functional as F
from torchvision.ops import masks_to_boxes

# - Matplotlib for plotting
import matplotlib.pyplot as plt

# - SAM2 specific modules
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# - Transformers from HuggingFace
from transformers import (
    AdamW,
    AutoModelForCausalLM,
    AutoProcessor,
    get_scheduler
)

# - Clustering algorithms from scikit-learn
from sklearn.cluster import DBSCAN

# - Progress bar utilities
from tqdm import tqdm

# - Supervision (sv) library
import supervision as sv

# Set environment variables for PyTorch
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# File: global_variables.py

import os
from pathlib import Path
import torch
import re

# Florence-2 settings
data_dir = Path("./snemi/")
raw_image_dir = data_dir / 'image_pngs'
seg_image_dir = data_dir / 'seg_pngs'
test_image_dir = data_dir / 'image_test_pngs'
max_output_token = 2048

CHECKPOINT = "microsoft/Florence-2-base-ft"
REVISION = 'refs/pr/6'

# Device settings (use GPU if available)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Label settings
label_num = 170

# Dataloader configuration
BATCH_SIZE = 5
NUM_WORKERS = 0

# Checkpoint directory for Florence-2 model
checkpoint_dir = './model_checkpoints/large_model/epoch_700'

# SAM2 settings
sam2_checkpoint = "./sam2_hiera_large.pt"
model_cfg = "./sam2_hiera_l.yaml"

# Overlap ratio used for SAM2
overlapp_ratio = 0.15

# checkpoint directory for SAM 2 model
finetuned_parameter_path = "./checkpoints/all/large_model_slice_7000.torch"

# Additional directories for SAM2
test_image_slice_dir = data_dir / "image_slice_test_pngs"
test_image_pred_dir = data_dir / "image_pred_test_pngs"

# Inference Parameters
num_corr = 10
num_steps = 10
num_file = 1

# Ensure directories exist for storing test image slices and predictions
os.makedirs(test_image_slice_dir, exist_ok=True)
os.makedirs(test_image_pred_dir, exist_ok=True)

# - Load Models
model = AutoModelForCausalLM.from_pretrained(checkpoint_dir, trust_remote_code=True).to(DEVICE)
processor = AutoProcessor.from_pretrained(checkpoint_dir, trust_remote_code=True)
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=DEVICE)
predictor = SAM2ImagePredictor(sam2_model)
predictor.model.load_state_dict(torch.load(finetuned_parameter_path))

# - check invalid bounding boxes
def validate_boxes(boxes):
    """Validate and filter out invalid bounding boxes."""
    valid_boxes = []
    for box in boxes:
        x_min, y_min, x_max, y_max = box[:4]
        if x_max > x_min and y_max > y_min:
            valid_boxes.append(box)
    return valid_boxes

# - Define Florence Inference
def flo_infer_bboxes(image_path, max_new_tokens, num_steps = 1):
    max_labels = 0
    max_bboxes = []
    max_image = []
    for i in tqdm(range(num_steps), desc="Florence2 Prediction", leave=False):
        image = cv2.imread(str(image_path))
        image = Image.fromarray(image)
        task = "<OD>"
        text = "<OD>"
        inputs = processor(
            text=text, 
            images=image, 
            return_tensors="pt"
        ).to(DEVICE)
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=max_new_tokens,
            num_beams=3)
        generated_text = processor.batch_decode(
            generated_ids, skip_special_tokens=False)[0]
        response = processor.post_process_generation(
            generated_text, 
            task=task, 
            image_size=image.size)
        detections = sv.Detections.from_lmm(
            sv.LMM.FLORENCE_2, response, resolution_wh=image.size)

        bounding_box_annotator = sv.BoxAnnotator(
            color_lookup=sv.ColorLookup.INDEX)
        label_annotator = sv.LabelAnnotator(
            color_lookup=sv.ColorLookup.INDEX)

        image = bounding_box_annotator.annotate(image, detections)
        image = label_annotator.annotate(image, detections)
        if len(response['<OD>']['bboxes']) > max_labels:
            max_labels = len(response['<OD>']['bboxes'])
            max_bboxes = response['<OD>']['bboxes']
            max_image = image
    return max_labels, max_bboxes, max_image

# - SAM2 Predict Segmentation
def predict_seg(
    img: np.ndarray,
    input_points: np.ndarray = None,
    point_labels: np.ndarray = None,
    input_boxes: np.ndarray = None,
    multimask_output: bool = False
) -> np.ndarray:
    torch.autocast(device_type='cpu', dtype=torch.bfloat16).__enter__()
    with torch.no_grad(): # prevent the net from caclulate gradient (more efficient inference)
            predictor.set_image(img) # image encoder
            masks, scores, logits = predictor.predict(  # prompt encoder + mask decoder
                point_coords=input_points,
                point_labels=point_labels,
                box = input_boxes,
                multimask_output=multimask_output)
            
    # - sort masks based on their scores (high-quality segmentation)
    shorted_masks = masks[np.argsort(scores[:,0])[::-1], :, :, :].astype(bool)
    seg_map = np.zeros_like(shorted_masks[0, 0, ...], dtype=np.uint8)
    occupancy_mask = np.zeros_like(shorted_masks[0, 0, ...],dtype=bool)

    # - add the masks one by one from high to low score
    for i in range(shorted_masks.shape[0]):
        if multimask_output:
            mask_lst = shorted_masks[i]
            score_lst = scores[i]
            score_rank = np.argsort(score_lst)[::-1]
            sorted_mask_lst = mask_lst[score_rank]
            for mask in sorted_mask_lst:
                if (mask*occupancy_mask).sum()/mask.sum() <= overlapp_ratio:
                    mask[occupancy_mask]=0
                    seg_map[mask]=i+1
                    occupancy_mask[mask]=1
                    break
        else:
             mask = shorted_masks[i, 0, ...]
             if (mask*occupancy_mask).sum()/mask.sum() <= overlapp_ratio:
                mask[occupancy_mask]=0
                seg_map[mask]=i+1
                occupancy_mask[mask]=1
        
    # - project back to RGB
    rgb_image = np.zeros((seg_map.shape[0], seg_map.shape[1], 3), dtype=np.uint8)
    for id_class in range(1,seg_map.max()+1):
        rgb_image[seg_map == id_class] = [np.random.randint(255), np.random.randint(255), np.random.randint(255)]

    return occupancy_mask, seg_map, rgb_image

# - Post Predict Segmentation
def post_predict_seg(
    img: np.ndarray,
    occupancy_mask,
    seg_map,
    input_points: np.ndarray = None,
    point_labels: np.ndarray = None,
    input_boxes: np.ndarray = None,
    multimask_output: bool = False,
) -> np.ndarray:
    torch.autocast(device_type='cpu', dtype=torch.bfloat16).__enter__()
    with torch.no_grad(): # prevent the net from caclulate gradient (more efficient inference)
            predictor.set_image(img) # image encoder
            masks, scores, logits = predictor.predict(  # prompt encoder + mask decoder
                point_coords=input_points,
                point_labels=point_labels,
                box = input_boxes,
                multimask_output=multimask_output)
            
    # - sort masks based on their scores (high-quality segmentation)
    if len(masks.shape) == 3:
        masks = masks.reshape((1, masks.shape[0], masks.shape[1], masks.shape[2]))
        scores = scores.reshape((1, len(scores)))
    shorted_masks = masks[np.argsort(scores[:,0])[::-1], :, :, :].astype(bool)
    seg_map = seg_map
    occupancy_mask = occupancy_mask
    max_id = np.max(seg_map) + 1

    # - add the masks one by one from high to low score
    for i in range(shorted_masks.shape[0]):
        if multimask_output:
            mask_lst = shorted_masks[i]
            score_lst = scores[i]
            score_rank = np.argsort(score_lst)[::-1]
            sorted_mask_lst = mask_lst[score_rank]
            for mask in sorted_mask_lst:
                if (mask*occupancy_mask).sum()/mask.sum() <= overlapp_ratio:
                    mask[occupancy_mask]=0
                    seg_map[mask]=max_id + i
                    occupancy_mask[mask]=1
                    break
        else:
             mask = shorted_masks[i, 0, ...]
             if (mask*occupancy_mask).sum()/mask.sum() <= overlapp_ratio:
                mask[occupancy_mask]=0
                seg_map[mask]=max_id + i
                occupancy_mask[mask]=1
        
    # - project back to RGB
    rgb_image = np.zeros((seg_map.shape[0], seg_map.shape[1], 3), dtype=np.uint8)
    for id_class in range(1,seg_map.max()+1):
        rgb_image[seg_map == id_class] = [np.random.randint(255), np.random.randint(255), np.random.randint(255)]

    return occupancy_mask, seg_map, rgb_image

# - Select Sparse Coordinates

def select_sparse_coordinates(occupancy_mask, 
    erode_kernel_size: tuple = (2, 2),
    erode_iterations: int = 10,
    dilate_kernel_size: tuple = (2, 2),
    dilate_iterations: int = 2,
    contour_area_threshold: int = 100
) -> np.ndarray:
    """
    Processes the input mask by performing erosion, dilation, contour filtering, 
    and marking centroids of the contours.
    
    Parameters:
    mask (np.ndarray): Input binary mask image.
    erode_kernel_size (tuple): Kernel size for erosion.
    erode_iterations (int): Number of iterations for erosion.
    dilate_kernel_size (tuple): Kernel size for dilation.
    dilate_iterations (int): Number of iterations for dilation.
    contour_area_threshold (int): Minimum area for contours to be considered valid.
    
    Returns:
    np.ndarray: Processed image with centroids marked.
    """
    
    # Step 3: Erode the mask
    img_inv = np.uint8(~occupancy_mask * 255)
    kernel = np.ones(erode_kernel_size, np.uint8)
    morphed_inv_mask = cv2.erode(img_inv, kernel, iterations=erode_iterations)

    # Step 4: Filter contours based on area threshold
    filt_inv_mask = np.zeros_like(morphed_inv_mask, dtype=np.uint8)
    contours, _ = cv2.findContours(morphed_inv_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cntr in contours:
        if cv2.contourArea(cntr) > contour_area_threshold:
            cv2.drawContours(filt_inv_mask, [cntr], -1, 255, thickness=cv2.FILLED)

    # Step 5: Dilate the filtered mask
    kernel = np.ones(dilate_kernel_size, np.uint8)
    morphed_filt_mask = cv2.dilate(filt_inv_mask, kernel, iterations=dilate_iterations)

    # Step 6: Find centroids of the remaining contours and collect coordinates
    centroids = []
    contours, _ = cv2.findContours(morphed_filt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cntr in contours:
        M = cv2.moments(cntr)
        if M['m00'] != 0:  # Avoid division by zero
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            centroids.append((cx, cy))

    # Convert list of centroids to a NumPy array
    return np.array(centroids)
    
# - flo + sam inference
def flo_sam_infer(image_path, num_corr = 5, num_steps = 10):
    num_labels, bboxes, image = flo_infer_bboxes(image_path, 2048, num_steps=num_steps)
    bboxes = validate_boxes(bboxes)
    img = cv2.imread(str(image_path))
    occupancy_mask, seg_map, rgb_image = predict_seg( img , None, None, input_boxes = bboxes)

    for i in tqdm(range(num_corr), desc='Continuous Correction', leave = False):
        false_coordinates = np.array(select_sparse_coordinates(occupancy_mask))
        false_labels = np.ones([false_coordinates.shape[0],])

        false_coordinates = false_coordinates.reshape((false_coordinates.shape[0], 1, false_coordinates.shape[1]))
        false_labels = np.ones([false_coordinates.shape[0], 1])

        occupancy_mask, seg_map, rgb_image = post_predict_seg( img, occupancy_mask, seg_map, input_points = false_coordinates, point_labels = false_labels, multimask_output=True)
    
    return rgb_image


class ImageSlicer:
    def __init__(self, image_path: str, slice_size: tuple, overlap_ratio: float, output_dir: str):
        self.image_path = image_path
        self.slice_size = slice_size
        self.overlap_ratio = overlap_ratio
        self.output_dir = output_dir

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load the image
        self.image = Image.open(self.image_path)
        self.image_width, self.image_height = self.image.size

    # - slicer padding problem
    def slice_image(self):
        # Calculate the step size based on the overlap ratio
        step_size_x = int(self.slice_size[0] * (1 - self.overlap_ratio))
        step_size_y = int(self.slice_size[1] * (1 - self.overlap_ratio))

        slice_paths = []
        slice_number = 0

        # Loop through the image dimensions to create overlapping slices
        for y in range(0, self.image_height - step_size_y, step_size_y):
            for x in range(0, self.image_width - step_size_x, step_size_x):
                # Define the box for cropping
                box = (x, y, min(x + self.slice_size[0], self.image_width), 
                             min(y + self.slice_size[1], self.image_height))
                slice_img = self.image.crop(box)

                # Save the slice with the coordinates in the filename
                slice_filename = f"{os.path.basename(self.image_path).split('.')[0]}_{box[0]}_{box[1]}_{box[2]}_{box[3]}.png"
                slice_path = os.path.join(self.output_dir, slice_filename)
                
                # Save the slice and append to the list of paths
                slice_img.save(slice_path)
                slice_paths.append(Path(slice_path))

                slice_number += 1

        return slice_paths

    def display_images_in_grid(self, image_paths):
        num_images = len(image_paths)
    
        # Determine the grid size: (cols, rows)
        cols = int(num_images**0.5)
        rows = (num_images // cols) + (num_images % cols > 0)
    
        # Create a figure with subplots
        fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    
        # Flatten axes array in case of multiple rows and columns
        axes = axes.flatten() if num_images > 1 else [axes]
    
        for i, image_path in enumerate(image_paths):
            img = Image.open(image_path)  # Open the image
            axes[i].imshow(img)           # Display the image
            axes[i].axis('off')           # Hide axes for better appearance
    
        # Remove any unused subplots (if the grid is larger than the number of images)
        for ax in axes[num_images:]:
            ax.remove()
    
        plt.tight_layout()
        plt.show()

    def convert_binary_mask(masks, image_seg, coords, total_shape):
        image_seg = image_seg[...,0]
        # get bounding boxes from the mask
        inds = np.unique(image_seg)[1:]  # load all indices
        xmin, ymin, xmax, ymax = coords

        # Prepare an empty mask with the total shape (e.g., 1024x1024)
        # Get binary masks and points
    
        masks = masks
        for ind in inds:
            total_mask = np.zeros(total_shape, dtype=np.uint8)
            mask = (image_seg == ind).astype(np.uint8)
            total_mask[ymin:ymax, xmin:xmax] = mask
            masks.append(total_mask)
        masks = np.array(masks)

        mask_tensor = torch.from_numpy(masks)

        boxes = masks_to_boxes(mask_tensor)

        return masks, boxes.numpy()

    def run_segmentation(self, slice_paths, total_shape, num_corr = 10, num_steps = 1):
        binary_masks = []
        for image_path in tqdm(slice_paths, desc = "Running Segmentation"):
            rgb_seg = flo_sam_infer(image_path, num_corr = num_corr, num_steps = num_steps)
            coords = image_path.stem.split("_")[1:]
            coords = [int(x) for x in coords]
            binary_masks = list(binary_masks)
            binary_masks, mask_boxes = convert_binary_mask(binary_masks, rgb_seg, coords, total_shape)
        combined_mask = np.sum(binary_masks, axis=0) > 0  # Sum all masks and threshold (1 if any overlap)
        plt.figure(figsize=(15, 15))
        plt.imshow(combined_mask, cmap='gray')
        plt.title('Combined Binary Mask')
        plt.axis('off')  # Hide axis labels and ticks
        plt.show()
        
        return combined_mask

def main():

    all_files = np.sort(os.listdir(test_image_dir))
    test_image_path_lst = np.array([test_image_dir / test_image_path for test_image_path in all_files])
    for i in range(num_file):
        image_sample_slicer = ImageSlicer(test_image_path_lst[i], (512, 512), 0.75, test_image_slice_dir)
        slice_lst = image_sample_slicer.slice_image()
        full_binary_mask = image_sample_slicer.run_segmentation(slice_lst, (1024, 1024), num_corr=10, num_steps=5)
        full_binary_image = Image.fromarray(full_binary_mask)
        img_id = re.findall(r'\d+', image_sample_slicer.image_path.stem)

        temp_path = test_image_pred_dir / ('seg' + img_id[0] + '.png')
        full_binary_image.save(temp_path)

if __name__ == "__main__":
    main()