import os
import argparse
import cv2
import json
import torch
import imageio
import numpy as np
import supervision as sv
from pathlib import Path
from supervision.draw.color import ColorPalette
import sys
sys.path.insert(0, './Grounded-SAM-2')

from utils.supervision_utils import CUSTOM_COLOR_MAP
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import torchvision 
from tqdm import tqdm
from collections import defaultdict

"""
Hyper parameters
"""
parser = argparse.ArgumentParser()
parser.add_argument('--grounding-model', default="IDEA-Research/grounding-dino-base")
parser.add_argument("--workdir", required=True)
parser.add_argument("--sam2-checkpoint", default="./preprocess/pretrained/sam2.1_hiera_large.pt")
parser.add_argument("--sam2-model-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
parser.add_argument("--output-dir", default="gsam2")
parser.add_argument("--input-dir", default="gpt_video")
parser.add_argument("--force-cpu", action="store_true")
parser.add_argument("--use-sport-specific", action="store_true", help="Use sport-specific dynamic objects for each video")
args = parser.parse_args()

GROUNDING_MODEL = args.grounding_model
INPUT_DIR = args.workdir
SAM2_CHECKPOINT = args.sam2_checkpoint
SAM2_MODEL_CONFIG = args.sam2_model_config
DEVICE = "cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"
OUTPUT_DIR = Path(args.output_dir)
USE_SPORT_SPECIFIC = args.use_sport_specific

# environment settings
# use bfloat16
torch.autocast(device_type=DEVICE, dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# build SAM2 image predictor
sam2_checkpoint = SAM2_CHECKPOINT
model_cfg = SAM2_MODEL_CONFIG
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=DEVICE)
sam2_predictor = SAM2ImagePredictor(sam2_model)

# build grounding dino from huggingface
model_id = GROUNDING_MODEL
processor = AutoProcessor.from_pretrained(model_id)
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(DEVICE)

CUSTOM_COLOR_MAP.pop(2)

def id_to_colors(id): # id to color
    rgb = np.zeros((3, ), dtype=np.uint8)
    for i in range(3):
        rgb[i] = id % 256
        id = id // 256
    return rgb

def get_sport_from_video(video_name):
    """Extract sport name from video name"""
    return video_name.split("_")[0]

def get_dyn_objs_from_gpt_output(json_path):
    """Extract dynamic objects from GPT analysis, focusing only on actually moving objects"""
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
            if "dynamic" in data:
                return data["dynamic"]
    except Exception as e:
        print(f"Error loading {json_path}: {str(e)}")
    return []

def filter_boxes_by_size(boxes, image_size, max_area_percent=10):
    """
    Filter bounding boxes by size relative to image area
    
    Args:
        boxes: Tensor of bounding boxes in format [x1, y1, x2, y2]
        image_size: Tuple of (height, width) of the image
        min_area_percent: Minimum area as a percentage of image area
        
    Returns:
        indices: List of indices of boxes to keep
    """
    image_height, image_width = image_size
    image_area = image_height * image_width
    max_area = (max_area_percent / 100) * image_area
    
    # Calculate box areas: (x2 - x1) * (y2 - y1)
    boxes_cpu = boxes.cpu()
    box_areas = (boxes_cpu[:, 2] - boxes_cpu[:, 0]) * (boxes_cpu[:, 3] - boxes_cpu[:, 1])
    
    # Create a mask for boxes to keep
    keep_indices = [i for i, area in enumerate(box_areas) if area <= max_area]
    
    return keep_indices

# Step 1: Group videos by sport and collect dynamic objects
videos = sorted([x for x in os.listdir(INPUT_DIR)])[::-1]
sport_dyn_objects = defaultdict(set)  # Dictionary to store dynamic objects by sport

print("Step 1: Collecting objects that are actually moving by sport...")
for video in tqdm(videos, desc="Collecting moving objects"):
        
    sport = get_sport_from_video(video)
    
    # Load dynamic objects from the video's GPT analysis
    text_prompt_file = os.path.join(INPUT_DIR, video, args.input_dir, "tags.json")
    if os.path.exists(text_prompt_file):
        # Get objects that GPT identified as actually moving in the frames
        moving_objs = get_dyn_objs_from_gpt_output(text_prompt_file)
        
        # Add objects showing movement to this sport's collection
        sport_dyn_objects[sport].update(set(moving_objs))

# Convert sets to sorted lists
sport_dyn_objects = {sport: sorted(list(objects)) for sport, objects in sport_dyn_objects.items()}

print(sport_dyn_objects)

# Step 2: Process each video with sport-specific dynamic objects
print("\nStep 2: Processing videos with sport-specific moving objects...")
for i, video in tqdm(enumerate(videos), desc="Processing videos"):

    work_dir = os.path.join(INPUT_DIR, video)
    image_dir_path = os.path.join(work_dir, "rgb")
    
    idx_to_id = [i for i in range(256*256*256)]
    np.random.shuffle(idx_to_id)  # mapping to randomize idx to id to get random color

    # Get sport-specific dynamic objects if enabled, otherwise use video-specific
    sport = get_sport_from_video(video)
    
    if USE_SPORT_SPECIFIC and sport in sport_dyn_objects and len(sport_dyn_objects[sport]) > 0:
        dyn_objs = sport_dyn_objects[sport]
        prompt_source = f"sport-specific moving objects ({sport})"
    else:
        # Load video-specific dynamic objects
        text_prompt_file = os.path.join(work_dir, args.input_dir, "tags.json")
        with open(text_prompt_file, "r") as f:
            dyn_objs = json.load(f)["dynamic"]
        prompt_source = "video-specific moving objects"
    
    text_input = ". ".join(dyn_objs) + "."
    print(f"Processing {video} with {prompt_source} prompt containing {len(dyn_objs)} moving objects: {dyn_objs}")
    
    # Create output directories
    output_path_vis = os.path.join(work_dir, OUTPUT_DIR, "vis")
    output_path_mask = os.path.join(work_dir, OUTPUT_DIR, "mask")
    os.makedirs(output_path_vis, exist_ok=True)
    os.makedirs(output_path_mask, exist_ok=True)
    
    # Save the prompt used for this video
    with open(os.path.join(work_dir, OUTPUT_DIR, "prompt_used.json"), "w") as f:
        json.dump({
            "moving_objects": dyn_objs,
            "prompt_source": prompt_source,
            "text_input": text_input
        }, f, indent=4)

    # Process each frame in the video
    for image_file in sorted(os.listdir(image_dir_path)):
        if not(image_file.endswith(".jpg") or image_file.endswith(".png")):
            continue

        full_image_path = os.path.join(image_dir_path, image_file)

        image_pil = Image.open(full_image_path).convert("RGB")
        image = np.array(image_pil)

        sam2_predictor.set_image(image)

        inputs = processor(images=image, text=text_input, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = grounding_model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=0.4,
            text_threshold=0.4,
            target_sizes=[image_pil.size[::-1]]
        )

        # Filter boxes by size (minimum 10% of image area)
        keep_indices = filter_boxes_by_size(
            results[0]["boxes"], 
            image_pil.size,  # (width, height)
            max_area_percent=70
        )
        
        # # Apply the filtering to all result elements
        if keep_indices:
            results[0]["boxes"] = results[0]["boxes"][keep_indices]
            results[0]["scores"] = results[0]["scores"][keep_indices]
            results[0]["labels"] = [results[0]["labels"][i] for i in keep_indices]
            if "text_labels" in results[0]:
                results[0]["text_labels"] = [results[0]["text_labels"][i] for i in keep_indices]


        # get the box prompt for SAM 2
        input_boxes = results[0]["boxes"].cpu().numpy()

        if input_boxes.shape[0] != 0:

            masks, scores, logits = sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_boxes,
                multimask_output=False,
            )

            """
            Post-process the output of the model to get the masks, scores, and logits for visualization
            """
            # convert the shape to (n, H, W)
            if masks.ndim == 4:
                masks = masks.squeeze(1)

            confidences = results[0]["scores"].cpu().numpy().tolist()
            class_names = results[0]["labels"]
            class_ids = np.array(list(range(len(class_names))))

            """
            Visualize image with supervision useful API
            """
            img = cv2.imread(full_image_path)
            detections = sv.Detections(
                xyxy=input_boxes,  # (n, 4)
                mask=masks.astype(bool),  # (n, h, w)
                class_id=class_ids,
                confidence=np.array(confidences)
            )

            assert(len(detections.class_id) > 0)

            nms_idx = torchvision.ops.nms(
                        torch.from_numpy(detections.xyxy).float(), 
                        torch.from_numpy(detections.confidence).float(), 
                        0.5
                    ).numpy().tolist()

            detections.xyxy = detections.xyxy[nms_idx]
            detections.class_id = detections.class_id[nms_idx]
            detections.confidence = detections.confidence[nms_idx]
            detections.mask = detections.mask[nms_idx]

            labels = [
                f"{class_names[id]} {confidence:.2f}"
                for id, confidence
                in zip(detections.class_id, detections.confidence)
            ]

            

            box_annotator = sv.BoxAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
            annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

            label_annotator = sv.LabelAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP), text_scale=1.0, text_thickness=2)
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

            mask_annotator = sv.MaskAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
            annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)

            masks = detections.mask
            labels = detections.class_id

            assert(np.sum(labels == -1) == 0) # check if any label == -1? concept graph has a bug with this

            color_mask = np.zeros(image.shape, dtype=np.uint8)

            obj_info_json = []

            #sort masks according to size
            mask_size = [np.sum(mask) for mask in masks]
            sorted_mask_idx = np.argsort(mask_size)[::-1]

            for idx in sorted_mask_idx: # render from largest to smallest
                
                mask = masks[idx]
                color_mask[mask] = id_to_colors(idx_to_id[idx])

                obj_info_json.append({
                    "id": idx_to_id[idx],
                    "label": class_names[labels[idx]],
                    "score": float(detections.confidence[idx]),
                })

            color_mask = cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)

            cv2.imwrite(os.path.join(output_path_vis, image_file), annotated_frame) # VISUALIZATION
            image_file = image_file.replace(".jpg", ".png")
            cv2.imwrite(os.path.join(output_path_mask, image_file), color_mask)
            with open(os.path.join(output_path_mask, image_file.replace(".png", ".json")), "w") as f:
                json.dump(obj_info_json, f)
        
        else:
            
            imageio.imwrite(os.path.join(output_path_vis, image_file), image) # VISUALIZATION
            image_file = image_file.replace(".jpg", ".png")
            cv2.imwrite(os.path.join(output_path_mask, image_file), np.zeros(image.shape, dtype=np.uint8))
            with open(os.path.join(output_path_mask, image_file.replace(".png", ".json")), "w") as f:
                json.dump([], f)
