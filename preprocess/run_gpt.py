import argparse
import os
import json
import glob
import io
from PIL import Image
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
import time

# Load OpenAI API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

def resize_image(image_path, max_size=512):
    """Resize image so that the longest dimension is max_size while preserving aspect ratio"""
    with Image.open(image_path) as img:
        # Get original dimensions
        width, height = img.size
        
        # Calculate new dimensions while maintaining aspect ratio
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
        
        # Resize image
        resized_img = img.resize((new_width, new_height), Image.LANCZOS)
        
        # Convert to bytes
        img_byte_arr = io.BytesIO()
        # Preserve original format if possible, fallback to JPEG
        format_name = img.format if img.format else 'JPEG'
        resized_img.save(img_byte_arr, format=format_name)
        
        return img_byte_arr.getvalue()

def sample_frames(img_files, interval=20):
    """Sample frames from the video with a fixed interval of 20 frames
    
    Args:
        img_files: List of image file paths
        interval: Interval between sampled frames (default: 20)
        max_samples: Maximum number of samples to return (default: None, meaning no limit)
    
    Returns:
        List of sampled image file paths
    """
    if len(img_files) <= interval:
        return img_files
    
    # Sample every 20 frames
    sampled_indices = list(range(0, len(img_files), interval))
    
    # Always include the first and last frame for context
    if len(img_files) - 1 not in sampled_indices and 0 not in sampled_indices:
        sampled_indices = [0] + sampled_indices
    
    if len(img_files) - 1 not in sampled_indices:
        sampled_indices.append(len(img_files) - 1)
    
    # Sort indices to maintain temporal order
    sampled_indices.sort()
    
    return [img_files[i] for i in sampled_indices]

def get_gpt4v_analysis(image_paths):
    """Get GPT-4V analysis of images with focus on identifying moving objects"""
    
    # Create messages with the system prompt
    messages = [
        {
            "role": "system", 
            "content": """You are a computer vision assistant specialized in analyzing video frames to identify objects that are ACTUALLY MOVING in the sequence.
            Your task is to identify categories of objects that are visibly in motion across the provided frames - NOT just objects that have the capacity to move.
            
            Focus only on objects that show evidence of movement or position changes between frames.
            
            For naming objects:
            1. Use singular form for all objects
            2. Use specific instance categories rather than general ones (e.g., 'man', 'woman', 'child' instead of 'person')
            3. Use lowercase format for consistency
            4. Only include objects that you can confidently determine are moving across the frames
            5. Use generic class names (if there are multiple man, just list 'man' once, not each individual man)"""
        }
    ]

    # Start the user message with text instructions
    user_message = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": """These are frames from a video. Analyze them and identify ONLY categories of objects that are ACTUALLY MOVING across these frames.
                
                Return ONLY a JSON response with the following structure:
                {
                    "dynamic": ["object1", "object2", "object3"],
                    "reasoning": "detailed explanation of how you identified movement between frames for each object type"
                }
                
                For the "dynamic" list:
                - Include ONLY object categories that show visible motion or position changes between frames
                - Use SINGULAR FORM for all objects (e.g., 'athlete' n~ot 'athletes')
                - For multiple instances of the same type, list the type only ONCE 
                - Keep terms simple and in lowercase
                - Do NOT include static objects or objects that merely have the capacity to move
                - If multiple instances of the same type are moving, list them individually (e.g., 'man', 'woman', 'child' instead of 'people')
                
                Be conservative - only include objects where you can clearly see evidence of motion across the frames."""
            }
        ]
    }
    
    # Add each image to the user's message content, resizing as needed
    for img_path in image_paths:
        # Resize the image and get binary data
        img_data = resize_image(img_path, max_size=512)
        
        # Encode to base64
        import base64
        base64_image = base64.b64encode(img_data).decode('utf-8')
        
        # Get the correct mime type
        extension = os.path.splitext(img_path)[1].lower()
        mime_type = "image/jpeg"  # Default
        if extension in ['.png']:
            mime_type = "image/png"
        elif extension in ['.jpg', '.jpeg']:
            mime_type = "image/jpeg"
        
        # Add to message
        user_message["content"].append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}
        })
    
    # Add the user message to messages
    messages.append(user_message)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            response_format={"type": "json_object"},
            max_tokens=1000
        )
        
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error getting GPT-4V analysis: {str(e)}")
        return {"error": str(e)}

def main():
    parser = argparse.ArgumentParser(description='Sample frames from videos and analyze with GPT-4V')
    parser.add_argument('--workdir', metavar='DIR', help='path to inputs', required=True)
    parser.add_argument('--output-dir', metavar='DIR', help='path to save outputs', default="gpt_video")
    parser.add_argument('--sample', type=int, default=20, help='sample every n frames')
    
    args = parser.parse_args()
    
    dataset_path = args.workdir
    videos = sorted(os.listdir(dataset_path))
    
    for video in tqdm(videos, desc="Processing videos"):

        start_time = time.time()
    
        img_dir = os.path.join(dataset_path, video, 'rgb')
        
        # Get all image files in the folder
        img_files = sorted(glob.glob(os.path.join(img_dir, '*.png')) + 
                          glob.glob(os.path.join(img_dir, '*.jpg')))
        
        if not img_files:
            print(f"No images found for video: {video}")
            continue
            
        # Sample frames from the video - use interval=20 instead of num_samples
        sampled_frames = sample_frames(img_files, interval=args.sample)
        
        # Get GPT-4V analysis of the sampled frames
        analysis_result = get_gpt4v_analysis(sampled_frames)
        
        assert("dynamic" in analysis_result)

        # Clean and standardize class names
        analysis_result["dynamic"] = [
            cls.lower().strip() for cls in analysis_result["dynamic"]
        ]
        # Remove duplicates while preserving order
        seen = set()
        analysis_result["dynamic"] = [
            cls for cls in analysis_result["dynamic"]
            if not (cls in seen or seen.add(cls))
        ]
        
        # Add metadata
        analysis_result["sampled_frames"] = [os.path.basename(frame) for frame in sampled_frames]
        
        # Create output directory
        output_path = os.path.join(dataset_path, video, args.output_dir)
        os.makedirs(output_path, exist_ok=True)
        
        # Save results
        with open(os.path.join(output_path, 'tags.json'), 'w') as f:
            json.dump(analysis_result, f, indent=4)

        end_time = time.time()
        time_taken = (end_time - start_time) / len(img_files)
        print(f"Time taken for {video}: {time_taken:.2f} seconds per frame")
        
        print(f"Processed {video}: Found {len(analysis_result['dynamic'])} moving objects")
        print(json.dumps(analysis_result, indent=4), flush=True)

if __name__ == "__main__":
    main()