# File: florence_sam_inference.py

# --------------------------------------------------------------
# Imports
# --------------------------------------------------------------

# System and OS packages
import os
import sys
import re
from pathlib import Path

# Numerical and scientific libraries
import numpy as np
import random

# PyTorch and related libraries
import torch
from torch.utils.data import Dataset, DataLoader

# Computer vision and image processing
import cv2
from PIL import Image
import torchvision.transforms.functional as F
from torchvision.ops import masks_to_boxes

# Plotting utilities
import matplotlib.pyplot as plt

# SAM2 specific modules
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Transformers from HuggingFace
from transformers import (
    AdamW,
    AutoModelForCausalLM,
    AutoProcessor,
    get_scheduler
)

# Clustering algorithms
from sklearn.cluster import DBSCAN

# Progress bar utilities
from tqdm import tqdm

# Supervision (sv) library
import supervision as sv

# Set environment variables for PyTorch
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# --------------------------------------------------------------
# Global Variables
# --------------------------------------------------------------

# Florence-2 model settings
data_dir = Path("./snemi/")
raw_image_dir = data_dir / 'image_pngs'
seg_image_dir = data_dir / 'seg_pngs'
test_image_dir = data_dir / 'image_test_pngs'
test_image_slice_dir = data_dir / "image_slice_test_pngs"
test_image_pred_dir = data_dir / "image_pred_test_pngs"

# Inference configuration
max_output_token = 2048
CHECKPOINT = "microsoft/Florence-2-base-ft"
REVISION = 'refs/pr/6'

# Device configuration (use GPU if available)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Dataset configuration
label_num = 170
BATCH_SIZE = 5
NUM_WORKERS = 0

# Checkpoint directories
checkpoint_dir = './model_checkpoints/large_model/epoch_700'
sam2_checkpoint = "./sam2_hiera_large.pt"
model_cfg = "./sam2_hiera_l.yaml"
finetuned_parameter_path = "./checkpoints/all/large_model_slice_7000.torch"

# Overlap ratios
overlapp_ratio = 0.1  # For SAM2
image_overlapp_ratio = 0.53  # For image slicing

# Inference parameters
num_corr = 30
num_steps = 15
num_file = 1

# Ensure necessary directories exist
os.makedirs(test_image_slice_dir, exist_ok=True)
os.makedirs(test_image_pred_dir, exist_ok=True)

# --------------------------------------------------------------
# Model Loading
# --------------------------------------------------------------

# Load Florence-2 model and processor
model = AutoModelForCausalLM.from_pretrained(checkpoint_dir, trust_remote_code=True).to(DEVICE)
processor = AutoProcessor.from_pretrained(checkpoint_dir, trust_remote_code=True)

# Build and load SAM2 model
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=DEVICE)
predictor = SAM2ImagePredictor(sam2_model)

# Load finetuned parameters into the SAM2 predictor
predictor.model.load_state_dict(torch.load(finetuned_parameter_path, map_location=DEVICE))


# --------------------------------------------------------------
# Bounding Box Validation
# --------------------------------------------------------------

def validate_boxes(boxes):
    """
    Validate and filter out invalid bounding boxes.
    A valid bounding box should have x_max > x_min and y_max > y_min.
    
    Args:
        boxes (list): List of bounding boxes in the form [x_min, y_min, x_max, y_max].
    
    Returns:
        valid_boxes (list): Filtered list of valid bounding boxes.
    """
    valid_boxes = []
    for box in boxes:
        x_min, y_min, x_max, y_max = box[:4]
        if x_max > x_min and y_max > y_min:
            valid_boxes.append(box)
    return valid_boxes


# --------------------------------------------------------------
# Florence Model Inference
# --------------------------------------------------------------

def flo_infer_bboxes(image_path, max_new_tokens, num_steps=1):
    """
    Perform inference using Florence-2 model to predict bounding boxes.

    Args:
        image_path (str): Path to the image file.
        max_new_tokens (int): Maximum number of new tokens to generate.
        num_steps (int): Number of prediction steps (for progressive refinement).

    Returns:
        max_labels (int): Maximum number of labels detected.
        max_bboxes (list): Detected bounding boxes.
        max_image (PIL.Image): Image annotated with bounding boxes.
    """
    max_labels = 0
    max_bboxes = []
    max_image = []
    
    for i in tqdm(range(num_steps), desc="Florence2 Prediction", leave=False):
        image = cv2.imread(str(image_path))
        image = Image.fromarray(image)
        task = "<OD>"  # Object detection task identifier
        inputs = processor(
            text=task, 
            images=image, 
            return_tensors="pt"
        ).to(DEVICE)
        
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=max_new_tokens,
            num_beams=3
        )
        
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        response = processor.post_process_generation(
            generated_text, 
            task=task, 
            image_size=image.size
        )
        
        detections = sv.Detections.from_lmm(
            sv.LMM.FLORENCE_2, response, resolution_wh=image.size
        )

        # Annotate bounding boxes on the image
        bounding_box_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
        label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX)
        image = bounding_box_annotator.annotate(image, detections)
        image = label_annotator.annotate(image, detections)

        if len(response['<OD>']['bboxes']) > max_labels:
            max_labels = len(response['<OD>']['bboxes'])
            max_bboxes = response['<OD>']['bboxes']
            max_image = image

    return max_labels, max_bboxes, max_image


# --------------------------------------------------------------
# SAM2 Predict Segmentation
# --------------------------------------------------------------

def predict_seg(
    img: np.ndarray,
    input_points: np.ndarray = None,
    point_labels: np.ndarray = None,
    input_boxes: np.ndarray = None,
    multimask_output: bool = False
) -> np.ndarray:
    """
    Perform segmentation prediction using SAM2 model.
    
    Args:
        img (np.ndarray): Input image as a numpy array.
        input_points (np.ndarray): Coordinates of input points for segmentation (optional).
        point_labels (np.ndarray): Labels for the input points (optional).
        input_boxes (np.ndarray): Input bounding boxes for segmentation (optional).
        multimask_output (bool): Flag to enable multi-mask output.
    
    Returns:
        occupancy_mask (np.ndarray): Occupancy mask showing which areas are covered by masks.
        seg_map (np.ndarray): Segmentation map with unique IDs for each mask.
        rgb_image (np.ndarray): Segmentation map projected back to an RGB image.
    """
    torch.autocast(device_type='cpu', dtype=torch.bfloat16).__enter__()

    with torch.no_grad():
        predictor.set_image(img)  # Set image for SAM2 predictor
        masks, scores, logits = predictor.predict(
            point_coords=input_points,
            point_labels=point_labels,
            box=input_boxes,
            multimask_output=multimask_output
        )

    # Sort masks based on their scores for high-quality segmentation
    if len(masks.shape) == 3:
        masks = masks.reshape((1, masks.shape[0], masks.shape[1], masks.shape[2]))
        scores = scores.reshape((1, len(scores)))
    sorted_masks = masks[np.argsort(scores[:, 0])[::-1], :, :, :].astype(bool)
    seg_map = np.zeros_like(sorted_masks[0, 0, ...], dtype=np.uint8)
    occupancy_mask = np.zeros_like(sorted_masks[0, 0, ...], dtype=bool)

    # Add masks one by one from high to low score
    for i in range(sorted_masks.shape[0]):
        if multimask_output:
            mask_lst = sorted_masks[i]
            score_lst = scores[i]
            score_rank = np.argsort(score_lst)[::-1]
            sorted_mask_lst = mask_lst[score_rank]
            for mask in sorted_mask_lst:
                if (mask * occupancy_mask).sum() / mask.sum() <= overlapp_ratio:
                    mask[occupancy_mask] = 0
                    seg_map[mask] = i + 1
                    occupancy_mask[mask] = 1
                    break
        else:
            mask = sorted_masks[i, 0, ...]
            if (mask * occupancy_mask).sum() / mask.sum() <= overlapp_ratio:
                mask[occupancy_mask] = 0
                seg_map[mask] = i + 1
                occupancy_mask[mask] = 1

    # Project segmentation back to RGB
    rgb_image = np.zeros((seg_map.shape[0], seg_map.shape[1], 3), dtype=np.uint8)
    for id_class in range(1, seg_map.max() + 1):
        rgb_image[seg_map == id_class] = [np.random.randint(255), np.random.randint(255), np.random.randint(255)]

    return occupancy_mask, seg_map, rgb_image


# --------------------------------------------------------------
# Post Segmentation Correction
# --------------------------------------------------------------

def post_predict_seg(
    img: np.ndarray,
    occupancy_mask,
    seg_map,
    input_points: np.ndarray = None,
    point_labels: np.ndarray = None,
    input_boxes: np.ndarray = None,
    multimask_output: bool = False,
) -> np.ndarray:
    
    if input_points.shape[0] == 0:
        rgb_image = np.zeros((seg_map.shape[0], seg_map.shape[1], 3), dtype=np.uint8)
        for id_class in range(1, seg_map.max() + 1):
            rgb_image[seg_map == id_class] = [np.random.randint(255), np.random.randint(255), np.random.randint(255)]

        return occupancy_mask, seg_map, rgb_image

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
    for id_class in range(1, seg_map.max() + 1):
        rgb_image[seg_map == id_class] = [np.random.randint(255), np.random.randint(255), np.random.randint(255)]

    return occupancy_mask, seg_map, rgb_image

# --------------------------------------------------------------
# Select Sparse Coordinates
# --------------------------------------------------------------

def select_sparse_coordinates(
    occupancy_mask: np.ndarray, 
    erode_kernel_size: tuple = (2, 2),
    erode_iterations: int = 10,
    dilate_kernel_size: tuple = (2, 2),
    dilate_iterations: int = 2,
    contour_area_threshold: int = 100
) -> np.ndarray:
    """
    Process the occupancy mask to find sparse coordinates (contours' centroids)
    after performing erosion, dilation, and contour filtering.

    Args:
        occupancy_mask (np.ndarray): Binary mask representing occupied areas.
        erode_kernel_size (tuple): Size of the kernel for erosion.
        erode_iterations (int): Number of iterations for erosion.
        dilate_kernel_size (tuple): Size of the kernel for dilation.
        dilate_iterations (int): Number of iterations for dilation.
        contour_area_threshold (int): Minimum contour area to be considered valid.

    Returns:
        np.ndarray: Array of centroids representing sparse coordinates.
    """
    
    # Step 1: Invert the occupancy mask and convert to uint8 format
    img_inv = np.uint8(~occupancy_mask * 255)
    
    # Step 2: Erode the inverted mask
    kernel = np.ones(erode_kernel_size, np.uint8)
    morphed_inv_mask = cv2.erode(img_inv, kernel, iterations=erode_iterations)

    # Step 3: Filter contours based on the area threshold
    filt_inv_mask = np.zeros_like(morphed_inv_mask, dtype=np.uint8)
    contours, _ = cv2.findContours(morphed_inv_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cntr in contours:
        if cv2.contourArea(cntr) > contour_area_threshold:
            cv2.drawContours(filt_inv_mask, [cntr], -1, 255, thickness=cv2.FILLED)

    # Step 4: Dilate the filtered mask
    kernel = np.ones(dilate_kernel_size, np.uint8)
    morphed_filt_mask = cv2.dilate(filt_inv_mask, kernel, iterations=dilate_iterations)

    # Step 5: Find centroids of the contours in the dilated mask
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


# --------------------------------------------------------------
# Combined Florence and SAM2 Inference
# --------------------------------------------------------------

def flo_sam_infer(image_path: str, num_corr: int = 5, num_steps: int = 10) -> np.ndarray:
    """
    Perform combined inference using Florence-2 for bounding boxes and SAM2 for segmentation.
    Applies corrections to the segmentation map based on sparse coordinates.

    Args:
        image_path (str): Path to the input image.
        num_corr (int): Number of correction iterations.
        num_steps (int): Number of inference steps for Florence-2.

    Returns:
        np.ndarray: The final RGB segmentation image after corrections.
    """
    
    # Step 1: Perform bounding box inference using Florence-2
    num_labels, bboxes, image = flo_infer_bboxes(image_path, 2048, num_steps=num_steps)
    bboxes = validate_boxes(bboxes)

    # Step 2: Read the image and perform SAM2 segmentation using the bounding boxes
    img = cv2.imread(str(image_path))
    occupancy_mask, seg_map, rgb_image = predict_seg(img, None, None, input_boxes=bboxes, multimask_output=True)

    # Step 3: Iteratively apply corrections based on sparse coordinates
    for i in tqdm(range(num_corr), desc='Continuous Correction', leave=False):
        false_coordinates = select_sparse_coordinates(occupancy_mask)

        # Reshape the coordinates and create point labels for SAM2 correction
        false_coordinates = false_coordinates.reshape((false_coordinates.shape[0], 1, 2))
        false_labels = np.ones((false_coordinates.shape[0], 1))

        # Apply SAM2 correction using the new coordinates
        occupancy_mask, seg_map, rgb_image = post_predict_seg(
            img, 
            occupancy_mask, 
            seg_map, 
            input_points=false_coordinates, 
            point_labels=false_labels, 
            multimask_output=True
        )
    
    return rgb_image


# --------------------------------------------------------------
# Image Slicer Class for Overlapping Image Slices
# --------------------------------------------------------------

class ImageSlicer:
    def __init__(self, image_path: str, slice_size: tuple, overlap_ratio: float, output_dir: str):
        """
        Initializes the ImageSlicer class.
        
        Args:
            image_path (str): Path to the image file to be sliced.
            slice_size (tuple): Size of the slices (width, height).
            overlap_ratio (float): Overlap ratio between slices.
            output_dir (str): Directory to save the sliced images.
        """
        self.image_path = image_path
        self.slice_size = slice_size
        self.overlap_ratio = overlap_ratio
        self.output_dir = output_dir

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load the image
        self.image = Image.open(self.image_path)
        self.image_width, self.image_height = self.image.size

    def slice_image(self) -> list:
        """
        Slices the image into overlapping regions and returns the file paths of the saved slices.
        
        Returns:
            slice_paths (list): List of file paths for the sliced images.
        """
        step_size_x = int(self.slice_size[0] * (1 - self.overlap_ratio))
        step_size_y = int(self.slice_size[1] * (1 - self.overlap_ratio))

        slice_paths = []

        # Loop through the image to create overlapping slices
        for y in range(0, self.image_height - step_size_y, step_size_y):
            for x in range(0, self.image_width - step_size_x, step_size_x):
                box = (x, y, min(x + self.slice_size[0], self.image_width), min(y + self.slice_size[1], self.image_height))
                slice_img = self.image.crop(box)

                # Create the filename for each slice
                slice_filename = f"{os.path.basename(self.image_path).split('.')[0]}_{box[0]}_{box[1]}_{box[2]}_{box[3]}.png"
                slice_path = os.path.join(self.output_dir, slice_filename)

                # Save the slice and add to the list
                slice_img.save(slice_path)
                slice_paths.append(Path(slice_path))

        return slice_paths

    def display_images_in_grid(self, image_paths: list):
        """
        Displays the sliced images in a grid layout.

        Args:
            image_paths (list): List of paths to sliced images.
        """
        num_images = len(image_paths)
        cols = int(np.sqrt(num_images))
        rows = (num_images // cols) + (num_images % cols > 0)

        fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
        axes = axes.flatten() if num_images > 1 else [axes]

        for i, image_path in enumerate(image_paths):
            img = Image.open(image_path)
            axes[i].imshow(img)
            axes[i].axis('off')

        for ax in axes[num_images:]:
            ax.remove()

        plt.tight_layout()
        plt.show()

    def convert_binary_mask(self, masks, image_seg, coords, total_shape) -> tuple:
        """
        Converts segmentation maps to binary masks and extracts bounding boxes.

        Args:
            masks (list): List of binary masks.
            image_seg (np.ndarray): Segmentation map.
            coords (list): Coordinates of the current slice.
            total_shape (tuple): Total shape of the combined mask (width, height).

        Returns:
            masks (np.ndarray): Updated masks array.
            boxes (np.ndarray): Array of bounding boxes.
        """
        image_seg = image_seg[..., 0]
        xmin, ymin, xmax, ymax = coords

        inds = np.unique(image_seg)[1:]  # Exclude the background
        for ind in inds:
            total_mask = np.zeros(total_shape, dtype=np.uint8)
            mask = (image_seg == ind).astype(np.uint8)
            total_mask[ymin:ymax, xmin:xmax] = mask
            masks.append(total_mask)

        masks = np.array(masks)
        mask_tensor = torch.from_numpy(masks)
        boxes = masks_to_boxes(mask_tensor)

        return masks, boxes.numpy()

    def run_segmentation(self, slice_paths: list, total_shape: tuple, num_corr: int = 10, num_steps: int = 1) -> np.ndarray:
        """
        Runs segmentation on all slices and combines the results into a full-sized binary mask.

        Args:
            slice_paths (list): List of file paths for the sliced images.
            total_shape (tuple): Total shape of the combined mask.
            num_corr (int): Number of correction iterations.
            num_steps (int): Number of inference steps.

        Returns:
            combined_mask (np.ndarray): Combined binary mask from all slices.
        """
        binary_masks = []
        for image_path in tqdm(slice_paths, desc="Running Segmentation", leave=False):
            rgb_seg = flo_sam_infer(image_path, num_corr=num_corr, num_steps=num_steps)
            coords = image_path.stem.split("_")[1:]
            coords = [int(x) for x in coords]
            binary_masks = list(binary_masks)
            binary_masks, mask_boxes = self.convert_binary_mask(binary_masks, rgb_seg, coords, total_shape)

        combined_mask = np.sum(binary_masks, axis=0) > 0  # Threshold combined masks
        plt.figure(figsize=(15, 15))
        plt.imshow(combined_mask, cmap='gray')
        plt.title('Combined Binary Mask')
        plt.axis('off')
        plt.show()

        return combined_mask


# --------------------------------------------------------------
# Main Function for Image Slicing and Segmentation
# --------------------------------------------------------------

def main():
    """
    Main function to slice images, run segmentation, and save results.
    """
    all_files = np.sort(os.listdir(test_image_dir))
    test_image_path_lst = np.array([test_image_dir / test_image_path for test_image_path in all_files])
    
    for i in tqdm(range(len(test_image_path_lst)), desc="Predict All Test Images"):
        # Create an ImageSlicer instance for each test image
        image_sample_slicer = ImageSlicer(test_image_path_lst[i], (512, 512), image_overlapp_ratio, test_image_slice_dir)

        # Slice the image
        slice_lst = image_sample_slicer.slice_image()

        # Run segmentation on the sliced images
        full_binary_mask = image_sample_slicer.run_segmentation(slice_lst, (1024, 1024), num_corr=num_corr, num_steps=num_steps)

        # Save the final binary mask
        full_binary_image = Image.fromarray(full_binary_mask)
        img_id = re.findall(r'\d+', image_sample_slicer.image_path.stem)
        temp_path = test_image_pred_dir / f'seg{img_id[0]}.png'
        full_binary_image.save(temp_path)


# --------------------------------------------------------------
# Entry Point for the Script
# --------------------------------------------------------------

if __name__ == "__main__":
    main()
