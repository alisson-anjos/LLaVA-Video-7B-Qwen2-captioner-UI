import os
import warnings
import numpy as np
import torch
import copy
import gc
from decord import VideoReader, cpu
from transformers import BitsAndBytesConfig

# LLaVA imports
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

warnings.filterwarnings("ignore")

# Configuration parameters
CONFIG = {
    # Video processing parameters
    'max_frames_num': 24,       # Maximum number of frames to extract. If force_sample is True, exactly this many frames will be used
    'force_sample': True,       # If True: always extract exactly max_frames_num frames evenly distributed across the video
    'fps': 1.0,                 # Target sampling rate. Only used if force_sample is False
                                # If False: use fps for sampling, but never exceed max_frames_num frames
    'batch_size': 8,            # Number of frames to process at once. Higher value = more memory usage but faster processing.
                                # Lower value = less memory usage but slower processing
    
    # Model parameters
    'model_path': "lmms-lab/LLaVA-Video-7B-Qwen2",  # Model to use
    'model_name': "llava_qwen",                      # Model name
    'torch_dtype': "bfloat16",                       # Torch data type
    'quantization_bits': 4,     # Quantization bits (4 or 8)
    
    # Generation parameters
    'max_new_tokens': 220,      # Maximum number of tokens to generate
    'temperature': 0.2,         # Generation temperature (0.0 - 1.0)
    'do_sample': True,         # Whether to use sampling
    
    # File handling parameters
    'input_folder': "/home/cseti/Data/Datasets/videos/Arcane/Cut Original/best_of/jinx/16x1360x768",         # Input folder path
    'output_csv': "/home/cseti/Data/Datasets/videos/Arcane/Cut Original/best_of/jinx/16x1360x768/captions.csv", # Output CSV file name
    
    # Prompt customization
    'prompt_template': """The video lasts for {video_time:.2f} seconds.
Please describe the scene in detail, focusing on:
- The viewing angle of the character (front view, side view, back view, or other angles)
- The character's position and orientation in the scene
- The character's movements and actions
- Any changes in the character's viewing angle during the video
- Other important visual details
Please be specific about the viewing perspective when describing the character."""
}

def get_sorted_video_files(folder_path):
    """Get video files from folder in numeric order"""
    # Supported video formats
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    
    # List files
    files = [f for f in os.listdir(folder_path) 
             if f.lower().endswith(video_extensions)]
    
    # Sort by numbers in filename
    try:
        files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
    except:
        files.sort()  # Fallback to alphabetical sort if numerical sort fails
    
    return [os.path.join(folder_path, f) for f in files]

def save_to_csv(filename, video_name, caption):
    """Save caption to CSV file"""
    import csv
    
    # Check if file exists
    file_exists = os.path.isfile(filename)
    
    with open(filename, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Write header for new file
        if not file_exists:
            writer.writerow(['video_name', 'caption'])
        
        writer.writerow([os.path.basename(video_name), caption])

def get_quantization_config(bits):
    """Create quantization configuration based on specified bits"""
    if bits not in [4, 8]:
        raise ValueError("Quantization must be either 4 or 8 bits!")
        
    if bits == 4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    else:  # 8 bit
        return BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16,
            bnb_8bit_use_double_quant=True
        )

def load_video(video_path, max_frames_num=CONFIG['max_frames_num'], 
               desired_fps=CONFIG['fps'], force_sample=CONFIG['force_sample']):
    """Load and sample frames from video"""
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3)), "", 0
    
    gc.collect()
    torch.cuda.empty_cache()
    
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frame_num = len(vr)
    video_fps = vr.get_avg_fps()
    video_time = total_frame_num / video_fps
    
    # Calculate frame skip based on desired fps
    skip_frames = int(video_fps / desired_fps)
    
    # Generate frame indices based on skip_frames
    frame_idx = list(range(0, total_frame_num, skip_frames))
    
    # If too many frames or force_sample is enabled
    if len(frame_idx) > max_frames_num or force_sample:
        indices = np.linspace(0, len(frame_idx) - 1, max_frames_num, dtype=int)
        frame_idx = [frame_idx[i] for i in indices]
    
    # Calculate timestamps
    frame_time = [idx/video_fps for idx in frame_idx]
    frame_time_str = ",".join(f"{t:.2f}s" for t in frame_time)
    
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    
    # Debug information
    print(f"Original video FPS: {video_fps:.2f}")
    print(f"Requested FPS: {desired_fps}")
    print(f"Frame skip: {skip_frames}")
    print(f"Number of selected frames: {len(frame_idx)}")
    
    del vr
    gc.collect()
    
    return spare_frames, frame_time_str, video_time

def main():
    # Initial memory cleanup
    gc.collect()
    torch.cuda.empty_cache()
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"Allocated GPU memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    else:
        print("CUDA is not available!")
        return

    # Load model with optimized settings
    print("\nLoading model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    quantization_config = get_quantization_config(CONFIG['quantization_bits'])

    try:
        tokenizer, model, image_processor, max_length = load_pretrained_model(
            CONFIG['model_path'], 
            None, 
            CONFIG['model_name'], 
            torch_dtype=CONFIG['torch_dtype'], 
            device_map="auto",
            attn_implementation="eager",
            quantization_config=quantization_config
        )

        model.eval()
        print("Model loaded successfully!")

        # Get input folder path
        input_folder = CONFIG['input_folder']
        
        if not os.path.exists(input_folder):
            print(f"The specified folder does not exist: {input_folder}")
            return
            
        # List video files
        video_files = get_sorted_video_files(input_folder)
        if not video_files:
            print("No video files found in the specified folder!")
            return
            
        print(f"\nFound {len(video_files)} video files")
        
        # Process videos
        for idx, video_path in enumerate(video_files, 1):
            print(f"\nProcessing: {idx}/{len(video_files)} - {os.path.basename(video_path)}")
            
            try:
                print("\nAllocated GPU memory before video processing:", 
                      f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB")
                
                print("\nProcessing video...")
                video, frame_time, video_time = load_video(
                    video_path, 
                    CONFIG['max_frames_num'], 
                    CONFIG['fps'], 
                    CONFIG['force_sample']
                )
                
                # Process frames in batches
                processed_frames = []
                for i in range(0, len(video), CONFIG['batch_size']):
                    batch = video[i:i+CONFIG['batch_size']]
                    processed_batch = image_processor.preprocess(batch, return_tensors="pt")["pixel_values"].to(device)
                    processed_frames.append(processed_batch.to(torch.bfloat16))
                    del batch
                    gc.collect()
                
                video = torch.cat(processed_frames, dim=0)
                video = [video]
                
                # Memory cleanup
                del processed_frames
                gc.collect()
                torch.cuda.empty_cache()
                
                print("Allocated GPU memory after video processing:", 
                      f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB")
                
                # Prepare prompt
                conv_template = "qwen_1_5"
                time_instruction = CONFIG['prompt_template'].format(
                    video_time=video_time
                )
                
                question = DEFAULT_IMAGE_TOKEN + f"\n{time_instruction}"
                
                conv = copy.deepcopy(conv_templates[conv_template])
                conv.append_message(conv.roles[0], question)
                conv.append_message(conv.roles[1], None)
                prompt_question = conv.get_prompt()
                
                # Generate
                input_ids = tokenizer_image_token(
                    prompt_question, 
                    tokenizer, 
                    IMAGE_TOKEN_INDEX, 
                    return_tensors="pt"
                ).unsqueeze(0).to(device)
                
                print("\nGenerating description...")
                
                with torch.inference_mode():
                    outputs = model.generate(
                        input_ids,
                        images=video,
                        modalities=["video"],
                        do_sample=CONFIG['do_sample'],
                        temperature=CONFIG['temperature'],
                        max_new_tokens=CONFIG['max_new_tokens'],
                        use_cache=True,
                    )
                
                description = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()
                print("\nGenerated video description:")
                print("-" * 50)
                print(description)
                print("-" * 50)
                
                # Save to CSV
                save_to_csv(CONFIG['output_csv'], video_path, description)
                
                # Memory cleanup
                del video, outputs
                gc.collect()
                torch.cuda.empty_cache()
                
                print("\nAllocated GPU memory after generation:", 
                      f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB")
                
            except Exception as e:
                print(f"\nError occurred while processing video: {str(e)}")
                print("Continuing with next video...")
                continue

            print(f"Processed: {os.path.basename(video_path)}")
            
        print(f"\nProcessing complete. Results saved to: {CONFIG['output_csv']}")

    except Exception as e:
        print(f"\nError occurred during execution: {str(e)}")

if __name__ == "__main__":
    main()