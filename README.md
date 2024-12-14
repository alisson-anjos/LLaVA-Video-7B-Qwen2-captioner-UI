- [LLaVA Video Captioner](#llava-video-captioner)
  - [Features](#features)
  - [Requirements](#requirements)
  - [Installation](#installation)
  - [Configuration](#configuration)
  - [Usage](#usage)
  - [Supported Video Formats](#supported-video-formats)
  - [Memory Optimization](#memory-optimization)
  - [Troubleshooting](#troubleshooting)
- [Caption refiner](#caption-refiner)
  - [Installation](#installation-1)
  - [Configuration](#configuration-1)
  - [Usage](#usage-1)


# LLaVA Video Captioner

This tool uses LLaVA (Large Language and Vision Assistant) to generate detailed descriptions of video content, with support for batch processing multiple videos.

## Features
- Batch video processing from a specified folder
- Frame sampling with configurable FPS
- Memory-optimized processing with batch support
- Customizable prompt templates
- CSV output for generated captions
- 4-bit or 8-bit model quantization support

## Requirements
- Python 3.10 or higher
- NVIDIA GPU with CUDA support
- Sufficient GPU memory (minimum 8GB recommended)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/cseti007/LLaVA-Video-7B-Qwen2-captioner.git
cd LLaVA-Video-7B-Qwen2-captioner
```

2. Create and activate a Python virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
.\venv\Scripts\activate  # On Windows
```

3. Install PyTorch with CUDA support:
```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

4. Install the LLaVA repository:
```bash
git clone https://github.com/LLaVA-VL/LLaVA-NeXT.git
cd LLaVA-NeXT
pip install -e .
cd ..
```

5. Install additional dependencies:
```bash
pip install decord transformers bitsandbytes
pip install einops av open_clip_torch
pip install 'accelerate>=0.26.0'
```

## Configuration

Edit the `CONFIG` dictionary in the script to customize:
- Input/output paths
- Video processing parameters
- Model settings
- Generation parameters
- Prompt template

Example configuration:
```python
CONFIG = {
   # Video processing parameters
    'max_frames_num': 24,       # Maximum number of frames to extract. If force_sample is True, exactly this many frames will be used
    'force_sample': True,       # If True: always extract exactly max_frames_num frames evenly distributed across the video
    'fps': 1.0,                 # Target sampling rate. Only used if force_sample is False
                                # If False: use fps for sampling, but never exceed max_frames_num frames
    'batch_size': 8,            # Number of frames to process at once. Higher value = more memory usage but faster processing.
                                # Lower value = less memory usage but slower processing
    
    # Model parameters
    'model_path': "lmms-lab/LLaVA-Video-7B-Qwen2",   # Model to use
    'model_name': "llava_qwen",                      # Model name
    'torch_dtype': "bfloat16",                       # Torch data type
    'quantization_bits': 4,                          # Quantization bits (4 or 8)
    
    # Generation parameters
    'max_new_tokens': 220,      # Maximum number of tokens to generate
    'temperature': 0.2,         # Generation temperature (0.0 - 1.0)
    'do_sample': True,          # Whether to use sampling
    
    # File handling parameters
    'input_folder': "path/to/videos",         # Input folder path
    'output_csv': "path/to/output.csv",       # Output CSV file name
}
```

## Usage

1. Update the configuration in the script:
   - Set `input_folder` to your videos directory
   - Set `output_csv` to your desired output location

2. Run the script:
```bash
python llava_video-video-captioner.py
```

The script will:
1. Process all videos in the input folder
2. Generate descriptions using the specified prompt template
3. Save results to the specified CSV file

## Supported Video Formats
- MP4
- AVI
- MOV
- MKV

## Memory Optimization

The script includes several memory optimization features:
- Batch processing of frames
- Garbage collection
- CUDA memory management
- 4-bit quantization (default)

You can adjust these parameters in the CONFIG dictionary:
- `batch_size`: Number of frames processed at once
- `max_frames_num`: Total frames to sample from each video
- `quantization_bits`: 4 or 8-bit quantization

## Troubleshooting

Common issues:

1. CUDA out of memory:
   - Reduce `batch_size`
   - Reduce `max_frames_num`
   - Use 4-bit quantization
   - Ensure no other processes are using GPU memory

2. Slow processing:
   - Increase `batch_size` if memory allows
   - Adjust `fps` for fewer frames
   - Consider using 8-bit quantization for better speed

3. Installation issues:
   - Make sure CUDA is properly installed
   - Check Python version compatibility
   - Try installing dependencies one by one to identify issues

4. Import errors:
   - Ensure all dependencies are installed
   - Check if virtual environment is activated
   - Verify CUDA compatibility with PyTorch version

# Caption refiner
This script is designed to automatically modify and refine text prompts stored in a CSV file using Ollama. It reads through the input CSV file in batches and processes each prompt according to specific config. The script takes each prompt and transforms it by adding a trigger word at the beginning, changing the subject of the prompt to be about a specific character (that can also be configured), and removing any video-related references while maintaining the core meaning. 

## Installation

1. Install prerequisites
```bash
pip install pandas langchain-core langchain-ollama
```

2. Install ollama
- Goto https://ollama.com
- Click on download, choose your OS then follow the instructions

3. Pull the ollama model
```bash
ollama pull llama3.2:3b
```

## Configuration

Edit the `CONFIG` dictionary in the script to customize:

Example configuration:
```python
# Configuration parameters
INPUT_CSV = "/path/to/file.csv"                # Path to input CSV file
OUTPUT_CSV = "/path/to/file.csv"              # Path to save the output CSV file
INPUT_COLUMN = "caption"         # Name of the column containing text to refine
OUTPUT_COLUMN = "refined_text"         # Name of the column where refined text will be saved
OLLAMA_MODEL = "llama3.2:3b"               # Name of the Ollama model to use
MAX_TOKENS = 200                      # Maximum number of tokens for the refined text
BATCH_SIZE = 10                       # Number of rows to process before saving progress

# Prompt templates
SYSTEM_PROMPT = """
You are an AI prompt engineer tasked with helping me modifying a list of automatically generated prompts.

Keep the original text but only do the following modifications:
- you responses should just be the prompt
- do not mention your task or the text itself
- add the following word to the start of each prompt: MYTRIGGERWORD
- modify each text so that ANNAMARIA is the main character in all of them, so use her name and since she's a woman, refer to her gender when necessary to make the sentences meaningful.
- remove references to video such as "the video begins" or "the video features" etc., but keep those sentences meaningful
- use only declarative sentences
"""
```
## Usage
1. Start the Ollama service if not running
2. Prepare your CSV file with required columns
3. Modify the script's configuration parameters (file paths, model name, etc.)
4. Run the script:
```bash
python ollama_caption_refinement-cseti.py
```
