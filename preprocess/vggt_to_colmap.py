import os
import argparse
import numpy as np
import torch
import glob
import struct
from scipy.spatial.transform import Rotation
import sys
from PIL import Image
import cv2
import requests
import tempfile
import collections
import math
import time
from torchvision import transforms as TF

sys.path.append("vggt/")

from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

def load_model(device=None):
    """Load and initialize the VGGT model."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model = VGGT.from_pretrained("facebook/VGGT-1B")

    # model = VGGT()
    # _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    # model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    
    model.eval()
    model = model.to(device)
    # model = model.to(torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16)

    return model, device

def load_and_preprocess_images(image_path_list, mode="crop"):
    """
    Function modified from VGGT: https://github.com/facebookresearch/vggt/blob/44b3afbd1869d8bde4894dd8ea1e293112dd5eba/vggt/utils/load_fn.py#L97
    A quick start function to load and preprocess images for model input.
    This assumes the images should have the same shape for easier batching, but our model can also work well with different shapes.

    Args:
        image_path_list (list): List of paths to image files
        mode (str, optional): Preprocessing mode, either "crop" or "pad".
                             - "crop" (default): Sets width to 518px and center crops height if needed.
                             - "pad": Preserves all pixels by making the largest dimension 518px
                               and padding the smaller dimension to reach a square shape.

    Returns:
        torch.Tensor: Batched tensor of preprocessed images with shape (N, 3, H, W)

    Raises:
        ValueError: If the input list is empty or if mode is invalid

    Notes:
        - Images with different dimensions will be padded with white (value=1.0)
        - A warning is printed when images have different shapes
        - When mode="crop": The function ensures width=518px while maintaining aspect ratio
          and height is center-cropped if larger than 518px
        - When mode="pad": The function ensures the largest dimension is 518px while maintaining aspect ratio
          and the smaller dimension is padded to reach a square shape (518x518)
        - Dimensions are adjusted to be divisible by 14 for compatibility with model requirements
    """
    # Check for empty list
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")
    
    # Validate mode
    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")

    images = []
    all_sizes = []
    shapes = set()
    to_tensor = TF.ToTensor()
    target_size = 518

    # First process all images and collect their shapes
    for image_path in image_path_list:

        # Open image
        img = Image.open(image_path)

        # If there's an alpha channel, blend onto white background:
        if img.mode == "RGBA":
            # Create white background
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            # Alpha composite onto the white background
            img = Image.alpha_composite(background, img)

        # Now convert to "RGB" (this step assigns white for transparent areas)
        img = img.convert("RGB")

        width, height = img.size
        
        if mode == "pad":
            # Make the largest dimension 518px while maintaining aspect ratio
            if width >= height:
                new_width = target_size
                new_height = round(height * (new_width / width) / 14) * 14  # Make divisible by 14
            else:
                new_height = target_size
                new_width = round(width * (new_height / height) / 14) * 14  # Make divisible by 14
        else:  # mode == "crop"
            # Original behavior: set width to 518px
            # new_width = target_size
            # # Calculate height maintaining aspect ratio, divisible by 14
            # new_height = round(height * (new_width / width) / 14) * 14
            assert(False)

        # Resize with new dimensions (width, height)
        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        img = to_tensor(img)  # Convert to tensor (0, 1)

        # Center crop height if it's larger than 518 (only in crop mode)
        if mode == "crop" and new_height > target_size:
            start_y = (new_height - target_size) // 2
            img = img[:, start_y : start_y + target_size, :]

        pad_top = 0
        pad_left = 0
        
        # For pad mode, pad to make a square of target_size x target_size
        if mode == "pad":
            h_padding = target_size - img.shape[1]
            w_padding = target_size - img.shape[2]

            if w_padding > 0:
                assert(False)
            
            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left
                
                # Pad with white (value=1.0)
                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )

        all_sizes.append((width, height, width / new_width, height / new_height, pad_top, pad_left))

        shapes.add((img.shape[1], img.shape[2]))
        images.append(img)

    # Check if we have different shapes
    # In theory our model can also work well with different shapes
    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes: {shapes}")
        # Find maximum dimensions
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)

        # Pad images if necessary
        padded_images = []
        for img in images:
            h_padding = max_height - img.shape[1]
            w_padding = max_width - img.shape[2]

            if h_padding > 0 or w_padding > 0:

                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )
            padded_images.append(img)
        images = padded_images

    images = torch.stack(images)  # concatenate images

    # Ensure correct shape when single image
    if len(image_path_list) == 1:
        # Verify shape is (1, C, H, W)
        if images.dim() == 3:
            images = images.unsqueeze(0)

    return images, all_sizes

def extrinsic_to_colmap_format(extrinsics):
    """Convert extrinsic matrices to COLMAP format (quaternion + translation)."""
    num_cameras = extrinsics.shape[0]
    quaternions = []
    translations = []
    
    for i in range(num_cameras):
        # VGGT's extrinsic is camera-to-world (R|t) format
        R = extrinsics[i, :3, :3]
        t = extrinsics[i, :3, 3]
        
        # Convert rotation matrix to quaternion
        # COLMAP quaternion format is [qw, qx, qy, qz]
        rot = Rotation.from_matrix(R)
        quat = rot.as_quat()  # scipy returns [x, y, z, w]
        quat = np.array([quat[3], quat[0], quat[1], quat[2]])  # Convert to [w, x, y, z]
        
        quaternions.append(quat)
        translations.append(t)
    
    return np.array(quaternions), np.array(translations)

def download_file_from_url(url, filename):
    """Downloads a file from a URL, handling redirects."""
    try:
        response = requests.get(url, allow_redirects=False)
        response.raise_for_status() 

        if response.status_code == 302:  
            redirect_url = response.headers["Location"]
            response = requests.get(redirect_url, stream=True)
            response.raise_for_status()
        else:
            response = requests.get(url, stream=True)
            response.raise_for_status()

        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded {filename} successfully.")
        return True

    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
        return False

def segment_sky(image_path, onnx_session, mask_filename=None):
    """
    Segments sky from an image using an ONNX model.
    """
    image = cv2.imread(image_path)

    result_map = run_skyseg(onnx_session, [320, 320], image)
    result_map_original = cv2.resize(result_map, (image.shape[1], image.shape[0]))

    # Fix: Invert the mask so that 255 = non-sky, 0 = sky
    # The model outputs low values for sky, high values for non-sky
    output_mask = np.zeros_like(result_map_original)
    output_mask[result_map_original < 32] = 255  # Use threshold of 32

    if mask_filename is not None:
        os.makedirs(os.path.dirname(mask_filename), exist_ok=True)
        cv2.imwrite(mask_filename, output_mask)
    
    return output_mask

def run_skyseg(onnx_session, input_size, image):
    """
    Runs sky segmentation inference using ONNX model.
    """
    import copy
    
    temp_image = copy.deepcopy(image)
    resize_image = cv2.resize(temp_image, dsize=(input_size[0], input_size[1]))
    x = cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB)
    x = np.array(x, dtype=np.float32)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    x = (x / 255 - mean) / std
    x = x.transpose(2, 0, 1)
    x = x.reshape(-1, 3, input_size[0], input_size[1]).astype("float32")

    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    onnx_result = onnx_session.run([output_name], {input_name: x})

    onnx_result = np.array(onnx_result).squeeze()
    min_value = np.min(onnx_result)
    max_value = np.max(onnx_result)
    onnx_result = (onnx_result - min_value) / (max_value - min_value)
    onnx_result *= 255
    onnx_result = onnx_result.astype("uint8")

    return onnx_result

def filter_and_prepare_points(predictions, conf_threshold, mask_sky=False, mask_black_bg=False, 
                             mask_white_bg=False, stride=1, prediction_mode="Depthmap and Camera Branch"):
    """
    Filter points based on confidence and prepare for COLMAP format.
    Implementation matches the conventions in the original VGGT code.
    """
    
    if "Pointmap" in prediction_mode:
        print("Using Pointmap Branch")
        if "world_points" in predictions:
            pred_world_points = predictions["world_points"]
            pred_world_points_conf = predictions.get("world_points_conf", np.ones_like(pred_world_points[..., 0]))
        else:
            print("Warning: world_points not found in predictions, falling back to depth-based points")
            pred_world_points = predictions["world_points_from_depth"]
            pred_world_points_conf = predictions.get("depth_conf", np.ones_like(pred_world_points[..., 0]))
    else:
        print("Using Depthmap and Camera Branch")
        pred_world_points = predictions["world_points_from_depth"]
        pred_world_points_conf = predictions.get("depth_conf", np.ones_like(pred_world_points[..., 0]))

    colors_rgb = predictions["images"] 
    
    S, H, W = pred_world_points.shape[:3]
    if colors_rgb.shape[:3] != (S, H, W):
        print(f"Reshaping colors_rgb from {colors_rgb.shape} to match {(S, H, W, 3)}")
        reshaped_colors = np.zeros((S, H, W, 3), dtype=np.float32)
        for i in range(S):
            if i < len(colors_rgb):
                reshaped_colors[i] = cv2.resize(colors_rgb[i], (W, H))
        colors_rgb = reshaped_colors
    
    colors_rgb = (colors_rgb * 255).astype(np.uint8)
    
    if mask_sky:
        print("Applying sky segmentation mask")
        try:
            import onnxruntime
         
            with tempfile.TemporaryDirectory() as temp_dir:
                print(f"Created temporary directory for sky segmentation: {temp_dir}")
                temp_images_dir = os.path.join(temp_dir, "images")
                sky_masks_dir = os.path.join(temp_dir, "sky_masks")
                os.makedirs(temp_images_dir, exist_ok=True)
                os.makedirs(sky_masks_dir, exist_ok=True)
                
                image_list = []
                for i, img in enumerate(colors_rgb):
                    img_path = os.path.join(temp_images_dir, f"image_{i:04d}.png")
                    image_list.append(img_path)
                    cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                
           
                skyseg_path = os.path.join(temp_dir, "skyseg.onnx")
                if not os.path.exists("skyseg.onnx"): 
                    print("Downloading skyseg.onnx...")
                    download_success = download_file_from_url(
                        "https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx", 
                        skyseg_path
                    )
                    if not download_success:
                        print("Failed to download skyseg model, skipping sky filtering")
                        mask_sky = False
                else:
            
                    import shutil
                    shutil.copy("skyseg.onnx", skyseg_path)
                
                if mask_sky:  
                    skyseg_session = onnxruntime.InferenceSession(skyseg_path)
                    sky_mask_list = []
                    
                    for img_path in image_list:
                        mask_path = os.path.join(sky_masks_dir, os.path.basename(img_path))
                        sky_mask = segment_sky(img_path, skyseg_session, mask_path)
           
                        if sky_mask.shape[0] != H or sky_mask.shape[1] != W:
                            sky_mask = cv2.resize(sky_mask, (W, H))
                        
                        sky_mask_list.append(sky_mask)
                    
                    sky_mask_array = np.array(sky_mask_list)
                    
                    sky_mask_binary = (sky_mask_array > 0.1).astype(np.float32)
                    pred_world_points_conf = pred_world_points_conf * sky_mask_binary
                    print(f"Applied sky mask, shape: {sky_mask_binary.shape}")
                
        except (ImportError, Exception) as e:
            print(f"Error in sky segmentation: {e}")
            mask_sky = False
    
    vertices_3d = pred_world_points.reshape(-1, 3)
    conf = pred_world_points_conf.reshape(-1)
    colors_rgb_flat = colors_rgb.reshape(-1, 3)

    

    if len(conf) != len(colors_rgb_flat):
        print(f"WARNING: Shape mismatch between confidence ({len(conf)}) and colors ({len(colors_rgb_flat)})")
        min_size = min(len(conf), len(colors_rgb_flat))
        conf = conf[:min_size]
        vertices_3d = vertices_3d[:min_size]
        colors_rgb_flat = colors_rgb_flat[:min_size]
    
    if conf_threshold == 0.0:
        conf_thres_value = 0.0
    else:
        conf_thres_value = np.percentile(conf, conf_threshold)
    
    print(f"Using confidence threshold: {conf_threshold}% (value: {conf_thres_value:.4f})")
    conf_mask = (conf >= conf_thres_value) & (conf > 1e-5)
    
    if mask_black_bg:
        print("Filtering black background")
        black_bg_mask = colors_rgb_flat.sum(axis=1) >= 16
        conf_mask = conf_mask & black_bg_mask
    
    if mask_white_bg:
        print("Filtering white background")
        white_bg_mask = ~((colors_rgb_flat[:, 0] > 240) & (colors_rgb_flat[:, 1] > 240) & (colors_rgb_flat[:, 2] > 240))
        conf_mask = conf_mask & white_bg_mask
    
    filtered_vertices = vertices_3d[conf_mask]
    filtered_colors = colors_rgb_flat[conf_mask]
    
    if len(filtered_vertices) == 0:
        print("Warning: No points remaining after filtering. Using default point.")
        filtered_vertices = np.array([[0, 0, 0]])
        filtered_colors = np.array([[200, 200, 200]])
    
    print(f"Filtered to {len(filtered_vertices)} points")
    
    points3D = []
    point_indices = {}
    image_points2D = [[] for _ in range(len(pred_world_points))]
    
    print(f"Preparing points for COLMAP format with stride {stride}...")
    
    total_points = 0
    for img_idx in range(S):
        for y in range(0, H, stride):
            for x in range(0, W, stride):
                flat_idx = img_idx * H * W + y * W + x
                
                if flat_idx >= len(conf):
                    continue
                
                if conf[flat_idx] < conf_thres_value or conf[flat_idx] <= 1e-5:
                    continue
                
                if mask_black_bg and colors_rgb_flat[flat_idx].sum() < 16:
                    continue
                
                if mask_white_bg and all(colors_rgb_flat[flat_idx] > 240):
                    continue
                
                point3D = vertices_3d[flat_idx]
                rgb = colors_rgb_flat[flat_idx]
                
                if not np.all(np.isfinite(point3D)):
                    continue
                
                point_hash = hash_point(point3D, scale=100)
                
                if point_hash not in point_indices:
                    point_idx = len(points3D)
                    point_indices[point_hash] = point_idx
                    
                    point_entry = {
                        "id": point_idx,
                        "xyz": point3D,
                        "rgb": rgb,
                        "error": 1.0,
                        "track": [(img_idx, len(image_points2D[img_idx]))]
                    }
                    points3D.append(point_entry)
                    total_points += 1
                else:
                    point_idx = point_indices[point_hash]
                    points3D[point_idx]["track"].append((img_idx, len(image_points2D[img_idx])))
                
                image_points2D[img_idx].append((x, y, point_indices[point_hash]))
    
    print(f"Prepared {len(points3D)} 3D points with {sum(len(pts) for pts in image_points2D)} observations for COLMAP")
    return points3D, image_points2D

def hash_point(point, scale=100):
    """Create a hash for a 3D point by quantizing coordinates."""
    quantized = tuple(np.round(point * scale).astype(int))
    return hash(quantized)

def write_colmap_cameras_txt(file_path, intrinsics, image_width, image_height):
    """Write camera intrinsics to COLMAP cameras.txt format."""
    with open(file_path, 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {len(intrinsics)}\n")
        
        for i, intrinsic in enumerate(intrinsics):
            camera_id = i + 1  # COLMAP uses 1-indexed camera IDs
            model = "PINHOLE" 
            
            fx = intrinsic[0, 0]
            fy = intrinsic[1, 1]
            cx = intrinsic[0, 2]
            cy = intrinsic[1, 2]
            
            f.write(f"{camera_id} {model} {image_width} {image_height} {fx} {fy} {cx} {cy}\n")

def write_colmap_images_txt(file_path, quaternions, translations, image_points2D, image_names):
    """Write camera poses and keypoints to COLMAP images.txt format."""
    with open(file_path, 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        
        num_points = sum(len(points) for points in image_points2D)
        avg_points = num_points / len(image_points2D) if image_points2D else 0
        f.write(f"# Number of images: {len(quaternions)}, mean observations per image: {avg_points:.1f}\n")
        
        for i in range(len(quaternions)):
            image_id = i + 1 
            camera_id = i + 1  
          
            qw, qx, qy, qz = quaternions[i]
            tx, ty, tz = translations[i]
            
            f.write(f"{image_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {camera_id} {image_names[i]}\n")
            
            points_line = " ".join([f"{x} {y} {point3d_id+1}" for x, y, point3d_id in image_points2D[i]])
            f.write(f"{points_line}\n")

def write_colmap_points3D_txt(file_path, points3D):
    """Write 3D points and tracks to COLMAP points3D.txt format."""
    with open(file_path, 'w') as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        
        avg_track_length = sum(len(point["track"]) for point in points3D) / len(points3D) if points3D else 0
        f.write(f"# Number of points: {len(points3D)}, mean track length: {avg_track_length:.4f}\n")
        
        for point in points3D:
            point_id = point["id"] + 1  
            x, y, z = point["xyz"]
            r, g, b = point["rgb"]
            error = point["error"]
            
            track = " ".join([f"{img_id+1} {point2d_idx}" for img_id, point2d_idx in point["track"]])
            
            f.write(f"{point_id} {x} {y} {z} {int(r)} {int(g)} {int(b)} {error} {track}\n")

def get_sports(base):
    """
    Get a list of sports from the base directory. Assumes directory names are formatted as "<sport>_<sport>_<other_info>".
    
    Args:
        base: Base path for the dataset
        
    Returns:
        List of sports
    """
    sports = set([])
    for dir in sorted(os.listdir(base)):
        sport = "_".join(dir.split("_")[:1])
        if sport not in sports:
            sports.add(sport)

    return sports

def sampling_strategy(mode, idx):
    
    """
    Adaptively samples at different rates based on total number of images to overcome OOM
    Args:
        mode: 0-5 -> higher number means more aggressive sampling
    """
    
    if mode == 0:   # use 100% frames
        return True
    elif mode == 1: # 75% frames
        if idx % 4 == 3:
            return False
        else:
            return True
    elif mode == 2: # 66% frames
        if idx % 3 == 2:
            return False
        else:
            return True
    elif mode == 3: # 50% frames
        if idx % 6 == 1 or idx % 6 == 3 or idx % 6 == 4:
            return False
        else:
            return True

def get_paths(base_path, sampling_mode, sport):
    """
    Get paths for images and masks based on the specified sport.
    
    Args:
        base_path: Base path for the dataset
        sport: Sport type (e.g., "volleyball")
        
    Returns:
        List of image paths
    """
    subdirs = [d for d in os.listdir(base_path) if (os.path.isdir(os.path.join(base_path, d)) and sport in d)]

    references = []

    subsubdirs = sorted([os.path.join(x, "rgb") for x in subdirs])
    
    for subdir in subsubdirs:
        for root, _, files in os.walk(os.path.join(base_path, subdir)):
            for i, file in enumerate(sorted(files)):
                
                if file.endswith(".jpg"):
                    references.append(os.path.join(subdir, file))
                if file.endswith(".png"):
                    references.append(os.path.join(subdir, file))

                if "cam" in subdir:
                    break

    total_images = len(references)
    print(f"Found {total_images} images in {sport} dataset.")

    references = []

    subsubdirs = sorted([os.path.join(x, "rgb") for x in subdirs])
    
    for subdir in subsubdirs:
        for root, _, files in os.walk(os.path.join(base_path, subdir)):
            for i, file in enumerate(sorted(files)):

                if "cam" not in subdir:   # dynamic cameras, use all frames

                    use_sample = sampling_strategy(sampling_mode, i)
                    if not use_sample:
                        continue
                    
                
                if file.endswith(".jpg"):
                    references.append(os.path.join(subdir, file))
                if file.endswith(".png"):
                    references.append(os.path.join(subdir, file))

                if "cam" in subdir:   # static camera, use first image
                    break

    references = sorted(references)[::-1]
    print(f"Found {len(references)} images in {sport} dataset.")

    return references

def get_camera_mapping(references):
    """
    Create a mapping from image names to camera IDs based on the camera number in the filename.
    
    Args:
        references: List of image paths
        
    Returns:
        Dictionary mapping image names to camera IDs
    """
    camera_mapping = {}
    checked = set()
    curr_cam_id = 1
    
    # First pass: identify all unique camera identifiers
    for id, ref in enumerate(references):
        # Extract just the filename from the path
        filename = ref.split("/")[0]
        
        # Split the filename and get the camera identifier (index 1)
        parts = filename.split('_')
        assert(len(parts))
        cam_id = parts[1]

        if cam_id not in checked:
            curr_cam_id = id+1
            checked.add(cam_id)

        camera_mapping[ref] = curr_cam_id

    return camera_mapping

def extract_camera_parameters(base, predictions, image_names, all_ratios, save_name):
    """
    Extract both camera intrinsics and extrinsics from VGGT predictions and save them
    in the same format as in run_hloc2.py.
    
    Args:
        base: Base path for the dataset
        predictions: Predictions from VGGT
        image_names: List of image names
        all_ratios: List of ratios to rescale intrinsics to accomodate for padding/cropping
        save_name: Name for saving the camera parameters
    """
    basepath_map = {}
    
    # Get extrinsics (camera-to-world matrices)
    extrinsics = predictions["extrinsic"]  # Shape: [N, 3, 4]
    intrinsics = predictions["intrinsic"]  # Shape: [N, 3, 3]
    
    for i, image_path in enumerate(image_names):
        # Get the base path (e.g., "cam1_volleyball_12345")
        parts = image_path.split(os.sep)
        if len(parts) >= 2:
            image_base = parts[0]  # The first part of the path
            image_name = parts[-1]  # The last part of the path (filename)
        else:
            # Fallback if path doesn't have expected structure
            image_base = "unknown"
            image_name = os.path.basename(image_path)
        
        if image_base not in basepath_map:
            basepath_map[image_base] = {
                "c2w": [],
                "K": [],
                "image_idx": []
            }
        
        # Get all images in this base directory
        rgb = os.path.join(base, image_base, "rgb")
        if os.path.exists(rgb):
            all_image_names = sorted(os.listdir(rgb))
            
            # Get the index of this image in the aligned directory

            idx = all_image_names.index(image_name)

            w2c_4x4 = np.eye(4, dtype=np.float32)
            w2c_4x4[:3, :4] = extrinsics[i]  # Copy the 3x4 part
            c2w_4x4 = np.linalg.inv(w2c_4x4)  # Invert to get world-to-camera

            w, h, w_ratio, h_ratio, top_pad, left_pad = all_ratios[i]

            intrinsic = intrinsics[i].copy()
            intrinsic[0,2] = intrinsic[0, 2] - left_pad
            intrinsic[1,2] = intrinsic[1, 2] - top_pad
            intrinsic[0, :] *= w_ratio
            intrinsic[1, :] *= h_ratio                
            basepath_map[image_base]["c2w"].append(c2w_4x4)
            basepath_map[image_base]["K"].append(intrinsic)
            basepath_map[image_base]["image_idx"].append(idx)
    
    # Save the camera parameters for each base path
    for basepath, data in basepath_map.items():
        # Convert lists to numpy arrays
        c2w = np.array(data["c2w"])
        K = np.array(data["K"])
        idxes = data["image_idx"]

        save_path = os.path.join(base, basepath, "vggt")
        os.makedirs(save_path, exist_ok=True)

        l = len(os.listdir(os.path.join(base, basepath, "rgb")))

        if "cam" in basepath: # stationary cameras
            c2w = np.repeat(c2w, l, axis=0)
            K = np.repeat(K, l, axis=0)
            valid = np.ones((l), dtype=bool)
        else:
            c2w = np.eye(4, dtype=np.float32)[None,:,:]
            c2w = np.repeat(c2w, l, axis=0)
            K = np.eye(3, dtype=np.float32)[None,:,:]
            K = np.repeat(K, l, axis=0)
            valid = np.zeros((l), dtype=bool)

            if l != len(data["c2w"]):
                print(f"Warning: {basepath} has {l} images, but only {len(data['c2w'])} camera parameters.")
                # continue

            for i, idx in enumerate(idxes):
                if idx < l:
                    c2w[idx] = data["c2w"][i]
                    K[idx] = data["K"][i]
                    valid[idx] = True

        np.savez_compressed(
            os.path.join(save_path, f"{save_name}.npz"),
            c2w=c2w,    # F x 4 x 4
            K=K,        # F x 3 x 3
            valid=valid,    # F
        )
        
        print(f"Saved camera parameters for {basepath} to {save_path}")

def process_sport_with_vggt(base, sport, device, model, sampling_mode=0):
    """
    Process all images for a specific sport using VGGT.
    
    Args:
        base: Base path for the dataset
        sport: Sport type (e.g., "volleyball")
        device: Torch device
        model: VGGT model
        sampling_mode: Sampling mode for image selection
        
    Returns:
        Tuple of (predictions, image_names)
    """
    # Get image paths for this sport
    references = get_paths(base, sampling_mode, sport)
    
    if not references:
        print(f"No images found for sport {sport}. Skipping.")
        return None, None

    full_paths = [os.path.join(base, x) for x in references]
        # Process images with VGGT
    images, all_ratios = load_and_preprocess_images(full_paths, mode="pad")
    images = images.to(device)
    
    print(f"Preprocessed images shape: {images.shape}")
    print("Running inference...")
    
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)
    
    print("Converting pose encoding to camera parameters...")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic
    
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)  # remove batch dimension
    
    # print("Computing 3D points from depth maps...")
    depth_map = predictions["depth"]  # (S, H, W, 1)
    world_points = unproject_depth_map_to_point_map(depth_map, predictions["extrinsic"], predictions["intrinsic"])
    predictions["world_points_from_depth"] = world_points
    
    # Prepare original images
    original_images = []
    for img_path in references:
        img = Image.open(os.path.join(base, img_path)).convert('RGB')
        original_images.append(np.array(img))
    
    predictions["original_images"] = original_images
    
    S, H, W = world_points.shape[:3]
    normalized_images = np.zeros((S, H, W, 3), dtype=np.float32)
    
    for i, img in enumerate(original_images):
        resized_img = cv2.resize(img, (W, H))
        normalized_images[i] = resized_img / 255.0
    
    predictions["images"] = normalized_images
    
    # Return the predictions and the original reference paths
    return predictions, references, all_ratios

def main():
    parser = argparse.ArgumentParser(description="Process images with VGGT and save camera parameters")
    parser.add_argument("--workdir", type=str, required=True,
                        help="Base directory for dataset")
    parser.add_argument("--vis_path", type=str, required=True,
                        help="vis directory to save camera parameters and COLMAP files for visualization")
    parser.add_argument("--save_colmap", action="store_true",
                        help="Whether to save camera parameters in COLMAP format for visualization")
    parser.add_argument("--conf_threshold", type=float, default=50.0, 
                        help="Confidence threshold (0-100%) for including points")
    parser.add_argument("--mask_sky", action="store_true",
                        help="Filter out points likely to be sky")
    parser.add_argument("--mask_black_bg", action="store_true",
                        help="Filter out points with very dark/black color")
    parser.add_argument("--mask_white_bg", action="store_true",
                        help="Filter out points with very bright/white color")
    parser.add_argument("--stride", type=int, default=10, 
                        help="Stride for point sampling in image (higher = fewer points)")
    parser.add_argument("--prediction_mode", type=str, default="Depthmap and Camera Branch",
                        choices=["Depthmap and Camera Branch", "Pointmap Branch"],
                        help="Which prediction branch to use")
    parser.add_argument("--save_name", type=str, default="camera_parameters",
                        help="Name for the saved camera parameters file")
    parser.add_argument("--sampling_mode", type=int, default=0,
                        help="Sampling mode for image selection (0-3, higher means more aggressive sampling)")
    
    args = parser.parse_args()
    
    BASE = args.workdir
    
    # Load VGGT model
    model, device = load_model()
    
    sports = get_sports(BASE)
    
    print(f"Working on sports: {sports}")

    for sport in sorted(sports):

        print(f"\nProcessing sport: {sport}")
        start_time = time.time()
        
        # Process images for this sport with VGGT
        predictions, references, all_ratios = process_sport_with_vggt(BASE, sport, device, model, args.sampling_mode)
        
        if predictions is None:
            continue
        
        # Create output directory for COLMAP files
        outputs = os.path.join(args.vis_path, "vggt_poses", f"{sport}")
        os.makedirs(outputs, exist_ok=True)
        
        # Save camera parameters in the same format as run_hloc2.py
        try:
            extract_camera_parameters(BASE, predictions, references, all_ratios, args.save_name)
        except Exception as e:
            print(f"Error extracting camera parameters: {e}")
            print(f"Skipping {sport} due to error.")
            continue

        end_time = time.time()
        time_taken = (end_time - start_time) / len(references)
        print(f"Processed {sport} in {time_taken:.2f} seconds")
        
        if args.save_colmap:
        
            # Also generate COLMAP files if needed
            print(f"Generating COLMAP files for {sport}...")
            
            colmap_dir = os.path.join(outputs, "colmap_vis")
            os.makedirs(colmap_dir, exist_ok=True)
            
            # Convert camera parameters to COLMAP format
            quaternions, translations = extrinsic_to_colmap_format(predictions["extrinsic"])
            
            # Filter and prepare points
            points3D, image_points2D = filter_and_prepare_points(
                predictions, 
                args.conf_threshold, 
                mask_sky=args.mask_sky, 
                mask_black_bg=args.mask_black_bg,
                mask_white_bg=args.mask_white_bg,
                stride=args.stride,
                prediction_mode=args.prediction_mode
            )
            
            height, width = predictions["depth"].shape[1:3]
            
            # Write COLMAP files
            write_colmap_cameras_txt(
                os.path.join(colmap_dir, "cameras.txt"), 
                predictions["intrinsic"], width, height)
            write_colmap_images_txt(
                os.path.join(colmap_dir, "images.txt"), 
                quaternions, translations, image_points2D, references)
            write_colmap_points3D_txt(
                os.path.join(colmap_dir, "points3D.txt"), 
                points3D)
        
            print(f"COLMAP files successfully written to {colmap_dir}")        
        print(f"Successfully processed sport: {sport}")
        
        #  clear all cuda memory
        torch.cuda.empty_cache()
        
        breakpoint()
        

if __name__ == "__main__":
    main()